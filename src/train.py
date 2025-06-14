import os
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch
import psutil
import time
from contextlib import contextmanager
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import recall_score, precision_score, f1_score


class GPUMonitor:
    """GPU memory monitoring utility"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = torch.cuda.current_device()
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if not self.enabled:
            return None
            
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'utilization_percent': (reserved / total) * 100
        }
    
    def print_gpu_stats(self, stage=""):
        """Print GPU memory statistics"""
        if not self.enabled:
            return
            
        info = self.get_gpu_memory_info()
        if info:
            print(f"\n=== GPU Memory Stats {stage} ===")
            print(f"Allocated: {info['allocated_gb']:.2f} GB")
            print(f"Reserved:  {info['reserved_gb']:.2f} GB")
            print(f"Free:      {info['free_gb']:.2f} GB")
            print(f"Total:     {info['total_gb']:.2f} GB")
            print(f"Usage:     {info['utilization_percent']:.1f}%")
            print("=" * 40)
    
    def log_peak_memory(self, stage=""):
        """Log peak memory usage and reset stats"""
        if not self.enabled:
            return
            
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**3
        
        print(f"\n=== Peak GPU Memory {stage} ===")
        print(f"Peak Allocated: {peak_allocated:.2f} GB")
        print(f"Peak Reserved:  {peak_reserved:.2f} GB")
        print("=" * 40)
        
        torch.cuda.reset_peak_memory_stats(self.device)
    
    @contextmanager
    def monitor_context(self, stage=""):
        if self.enabled:
            self.print_gpu_stats(f"[Before {stage}]")
            torch.cuda.reset_peak_memory_stats(self.device)
            start_time = time.time()
        
        try:
            yield
        finally:
            if self.enabled:
                end_time = time.time()
                self.log_peak_memory(f"[Peak during {stage}]")
                self.print_gpu_stats(f"[After {stage}]")
                print(f"Duration: {end_time - start_time:.2f} seconds\n")


class EmotionClassifierTrainer:
    def __init__(self, config_path="config.yaml"):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self._setup_environment()
        set_seed(self.config['seed'])
        
        self.gpu_monitor = GPUMonitor(
            enabled=self.config.get('hardware', {}).get('monitor_gpu', True)
        )
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_environment(self):
        """Setup CUDA environment"""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config['hardware']['cuda_device'])
    
    def load_data(self):
        """Load and prepare the dataset"""
        dataset_config = self.config['dataset']
        
        train = load_dataset(dataset_config['name'], dataset_config['language'], split="train")
        val = load_dataset(dataset_config['name'], dataset_config['language'], split="dev")
        test = load_dataset(dataset_config['name'], dataset_config['language'], split="test")
        
        emotion_cols = dataset_config['emotion_columns']
        num_labels = len(emotion_cols)
        
        def create_labels(examples):
            labels = []
            for i in range(len(examples['text'])):
                label = [float(examples[col][i]) for col in emotion_cols]
                labels.append(label)
            examples['label'] = labels
            return examples

        train = train.map(create_labels, batched=True)
        val = val.map(create_labels, batched=True)
        test = test.map(create_labels, batched=True)
        
        return train, val, test, emotion_cols, num_labels

    def setup_model_and_tokenizer(self, num_labels):
        """Setup model and tokenizer"""
        model_config = self.config['model']
        lora_config_dict = self.config['lora']
        
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'], padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            'trust_remote_code': model_config['trust_remote_code'],
            'torch_dtype': getattr(torch, model_config['torch_dtype']),
        }

        if model_config.get('use_quantization', False):
            quant_config = model_config['quantization_config']
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant_config['load_in_4bit'],
                bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
                bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
                bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_config['name'], 
            use_cache=model_config['use_cache'], 
            num_labels=num_labels, 
            problem_type=model_config['problem_type'], 
            **model_kwargs
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        
        lora_config = LoraConfig(
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            target_modules=lora_config_dict['target_modules'],
            lora_dropout=lora_config_dict['lora_dropout'],
            bias=lora_config_dict['bias'],
            task_type=TaskType.SEQ_CLS,
            modules_to_save=lora_config_dict['modules_to_save'],
        )

        if model_config.get('use_quantization', False):
            model = prepare_model_for_kbit_training(model)
        else:
            model.gradient_checkpointing_enable()
        
            for name, param in model.named_parameters():
                if (
                    "self_attn" not in name
                    and "output.dense" not in name
                    and "intermediate.dense" not in name
                    and "score" not in name 
                ):
                    param.requires_grad = False

            for name, param in model.named_parameters():
                if "lora_" in name or "score" in name:
                    param.requires_grad = True

        
        model = get_peft_model(model, lora_config)
        
        return model, tokenizer

    def tokenize_datasets(self, train, val, test, tokenizer):
        """Tokenize all datasets"""
        def tokenize_func(example):
            return tokenizer(
                example["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.config['dataset']['max_length'],
            )

        train_ds = train.map(tokenize_func, batched=True, desc="Tokenizing train data")
        eval_ds = val.map(tokenize_func, batched=True, desc="Tokenizing eval data")
        test_ds = test.map(tokenize_func, batched=True, desc="Tokenizing test data")
        
        return train_ds, eval_ds, test_ds

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        y_pred = predictions > self.config['evaluation']['threshold']

        results = {}
        for average in ['micro', 'macro']:
            results[f'{average}_recall'] = recall_score(labels, y_pred, average=average, zero_division=0)
            results[f'{average}_precision'] = precision_score(labels, y_pred, average=average, zero_division=0)
            results[f'{average}_f1'] = f1_score(labels, y_pred, average=average, zero_division=0)
        return results

    def create_training_args(self):
        """Create training arguments from config"""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config['output_dir'],
            eval_strategy=training_config['eval_strategy'],
            save_strategy=training_config['save_strategy'],
            logging_strategy=training_config['logging_strategy'],
            logging_steps=training_config['logging_steps'],
            learning_rate=training_config['learning_rate'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            warmup_ratio=training_config['warmup_ratio'],
            bf16=training_config['bf16'],
            fp16=training_config['fp16'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            group_by_length=training_config['group_by_length'],
            max_grad_norm=training_config['max_grad_norm'],
            weight_decay=training_config['weight_decay'],
        )

    def train_model(self):
        """Main training function"""
        train, val, test, emotion_cols, num_labels = self.load_data()
        
        model, tokenizer = self.setup_model_and_tokenizer(num_labels)
        model.print_trainable_parameters()
        
        train_ds, eval_ds, test_ds = self.tokenize_datasets(train, val, test, tokenizer)
        
        data_collator = DataCollatorWithPadding(
            tokenizer, 
            pad_to_multiple_of=self.config['hardware']['pad_to_multiple_of']
        )
        
        training_args = self.create_training_args()
        
        trainer = Trainer(
            model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        with self.gpu_monitor.monitor_context("Training"):
            trainer.train()
        
        self.evaluate_test_set(trainer, test_ds, self.config['evaluation']['threshold'], emotion_cols)
        
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        with open(f"{training_args.output_dir}/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
                    
        print(f"Model and configuration saved to {training_args.output_dir}")
        

    def evaluate_test_set(self, trainer, test_ds, threshold, emotion_cols):
        """Evaluate model on test set"""
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_predictions = trainer.predict(test_ds)
        test_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).numpy()
        
        test_pred_labels = (test_probs > threshold).astype(int)
        true_test_labels = test_predictions.label_ids
        
        print("\nTest Results:")
        for average in ['micro', 'macro']:
            recall = recall_score(true_test_labels, test_pred_labels, average=average, zero_division=0)
            precision = precision_score(true_test_labels, test_pred_labels, average=average, zero_division=0)
            f1 = f1_score(true_test_labels, test_pred_labels, average=average, zero_division=0)
            print(f'{average.upper()} - Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}')

        print(f"\nPer-class Results:")
        class_recall = recall_score(true_test_labels, test_pred_labels, average=None, zero_division=0)
        class_precision = precision_score(true_test_labels, test_pred_labels, average=None, zero_division=0)
        class_f1 = f1_score(true_test_labels, test_pred_labels, average=None, zero_division=0)

        for i, emotion in enumerate(emotion_cols):
            print(f'{emotion.upper()}: Recall: {class_recall[i]:.4f}, Precision: {class_precision[i]:.4f}, F1: {class_f1[i]:.4f}')

        print(f"\nClass distribution in test set:")
        for i, emotion in enumerate(emotion_cols):
            true_count = int(true_test_labels[:, i].sum())
            pred_count = int(test_pred_labels[:, i].sum())
            total = len(true_test_labels)
            print(f'{emotion.upper()}: True: {true_count}/{total} ({true_count/total:.1%}), Predicted: {pred_count}/{total} ({pred_count/total:.1%})')


def main():
    parser = argparse.ArgumentParser(description='Train emotion classification model')
    parser.add_argument('--config', type=str, default='../config/config.yaml', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    trainer = EmotionClassifierTrainer(args.config)
    trainer.train_model()


if __name__ == "__main__":
    main()