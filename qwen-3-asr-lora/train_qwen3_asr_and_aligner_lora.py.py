import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# Set random seed
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# SECTION 1: DUMMY DATASET PREPARATION
# ==============================================================================

def prepare_dummy_datasets():
    """
    Creates dummy dataset files for ASR and Aligner to make this script immediately executable.
    Replace 'samples/sample1.wav' with paths to your actual 16kHz audio files.
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # 1. Generate a dummy 16kHz sine wave audio file if none exists
    import soundfile as sf
    import numpy as np

    sample_audio_path = "samples/sample1.wav"
    if not os.path.exists(sample_audio_path):
        sample_rate = 16000
        duration = 3.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(sample_audio_path, tone.astype(np.float32), sample_rate)

    # 2. Dataset manifest for Qwen3-ASR 1.7B
    asr_data = [
        {"audio": sample_audio_path, "sentence": "hello world fine tuning qwen speech recognition"},
        {"audio": sample_audio_path, "sentence": "nvidia a6000 fine tuning lora pipeline"},
    ]
    with open("data/asr_train.json", "w", encoding="utf-8") as f:
        json.dump(asr_data, f, indent=2)

    # 3. Dataset manifest for Qwen3-ForcedAligner 0.6B
    # target_timestamps structure: [[start_sec, end_sec], ...] per token/word
    aligner_data = [
        {
            "audio": sample_audio_path,
            "text": "hello world",
            "text_tokens": [14990, 1879],  # Example token IDs from tokenizer
            "target_timestamps": [[0.1, 0.6], [0.65, 1.2]]
        }
    ]
    with open("data/aligner_train.json", "w", encoding="utf-8") as f:
        json.dump(aligner_data, f, indent=2)

    print("[✔] Dummy datasets and audio manifests initialized in ./data/")


# ==============================================================================
# SECTION 2: QWEN3-ASR-1.7B FINE-TUNING (AUTOREGRESSIVE)
# ==============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def train_asr_lora():
    print("\n" + "="*50)
    print("STARTING QWEN3-ASR-1.7B LoRA FINE-TUNING")
    print("="*50)

    model_id = "Qwen/Qwen3-ASR-1.7B"

    # 1. Load Processor & Base Model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Attach LoRA Adapter
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Preprocess Dataset
    raw_dataset = Dataset.from_json("data/asr_train.json").cast_column("audio", Audio(sampling_rate=16000))

    def prepare_asr_batch(batch):
        audio = batch["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    processed_dataset = raw_dataset.map(prepare_asr_batch, remove_columns=raw_dataset.column_names)

    # 4. Define Metrics (WER)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 5. Trainer Configuration
    training_args = TrainingArguments(
        output_dir="./qwen3_asr_lora_output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_steps=10,  # Increase for actual training
        bf16=True,
        logging_steps=2,
        save_strategy="steps",
        save_steps=10,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False
    trainer.train()

    # Save final LoRA Adapter
    output_dir = "./qwen3_asr_lora_final"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[✔] Qwen3-ASR LoRA weights successfully saved to {output_dir}")


# ==============================================================================
# SECTION 3: QWEN3-FORCEDALIGNER-0.6B FINE-TUNING (NON-AUTOREGRESSIVE)
# ==============================================================================

@dataclass
class ForcedAlignerCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        batch["text_tokens"] = torch.tensor([f["text_tokens"] for f in features], dtype=torch.long)
        batch["target_timestamps"] = torch.tensor([f["target_timestamps"] for f in features], dtype=torch.float32)
        return batch


class ForcedAlignerTrainer(Trainer):
    """Custom Trainer implementing Smooth L1 Loss for boundary timestamp regression."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_features=inputs["input_features"],
            text_tokens=inputs["text_tokens"]
        )
        
        # Access alignment output timestamps (adjust attribute based on raw head structure)
        pred_timestamps = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        target_timestamps = inputs["target_timestamps"]

        loss_fct = nn.SmoothL1Loss()
        loss = loss_fct(pred_timestamps, target_timestamps)

        return (loss, outputs) if return_outputs else loss


def train_forced_aligner_lora():
    print("\n" + "="*50)
    print("STARTING QWEN3-FORCEDALIGNER-0.6B LoRA FINE-TUNING")
    print("="*50)

    model_id = "Qwen/Qwen3-ForcedAligner-0.6B"

    # 1. Load Processor & Base Model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Configure LoRA for Feature Extractor & Align Projection
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Preprocess Dataset
    raw_dataset = Dataset.from_json("data/aligner_train.json").cast_column("audio", Audio(sampling_rate=16000))

    def prepare_aligner_batch(batch):
        audio = batch["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        batch["input_features"] = inputs.input_features[0]
        batch["text_tokens"] = batch["text_tokens"]
        batch["target_timestamps"] = batch["target_timestamps"]
        return batch

    processed_dataset = raw_dataset.map(prepare_aligner_batch, remove_columns=raw_dataset.column_names)

    # 4. Trainer Configuration
    training_args = TrainingArguments(
        output_dir="./qwen3_aligner_lora_output",
        per_device_train_batch_size=16,
        learning_rate=3e-4,
        max_steps=10,  # Increase for actual training
        bf16=True,
        logging_steps=2,
        save_strategy="steps",
        save_steps=10,
        report_to="tensorboard",
    )

    trainer = ForcedAlignerTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=ForcedAlignerCollator(processor=processor),
    )

    trainer.train()

    # Save final LoRA Adapter
    output_dir = "./qwen3_aligner_lora_final"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[✔] Qwen3-ForcedAligner LoRA weights successfully saved to {output_dir}")


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

if __name__ == "__main__":
    # Step 1: Create dummy manifests
    prepare_dummy_datasets()

    # Step 2: Fine-tune ASR
    train_asr_lora()

    # Step 3: Fine-tune ForcedAligner
    train_forced_aligner_lora()

    print("\n[✔] ALL FINE-TUNING PIPELINES COMPLETED SUCCESSFULLY!")