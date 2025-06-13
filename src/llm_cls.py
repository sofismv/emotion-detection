import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dataclasses import asdict, dataclass
import evaluate
import numpy as np
import simple_parsing
import torch
import wandb
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
import optuna
from sklearn.metrics import recall_score, precision_score, f1_score

def load_data():
    """Load and prepare the dataset"""
    train = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", split="train")
    val = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", split="dev")
    test = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", split="test")
    
    emotion_cols = ['anger', 'fear', 'joy', 'disgust', 'sadness', 'surprise']
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

def setup_model_and_tokenizer(model_name, num_labels, trial):
    """Setup model and tokenizer with trial parameters"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, use_cache=False, num_labels=num_labels, problem_type="multi_label_classification", **model_kwargs
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    lora_r = trial.suggest_categorical("lora_r", [8, 16])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.05, 0.1])
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model, tokenizer

def tokenize_datasets(train, val, test, tokenizer):
    """Tokenize all datasets"""
    def tokenize_func(example):
        return tokenizer(
            example["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )

    train_ds = train.map(tokenize_func, batched=True, desc="Tokenizing train data")
    eval_ds = val.map(tokenize_func, batched=True, desc="Tokenizing eval data")
    test_ds = test.map(tokenize_func, batched=True, desc="Tokenizing test data")
    
    return train_ds, eval_ds, test_ds

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    y_pred = predictions > 0.5

    results = {}
    for average in ['micro', 'macro']:
        results[f'{average}_recall'] = recall_score(labels, y_pred, average=average, zero_division=0)
        results[f'{average}_precision'] = precision_score(labels, y_pred, average=average, zero_division=0)
        results[f'{average}_f1'] = f1_score(labels, y_pred, average=average, zero_division=0)
    return results

def find_best_threshold(trainer, val_dataset, thresholds=np.arange(0.1, 0.9, 0.05)):
    """Find the best threshold for predictions"""
    predictions = trainer.predict(val_dataset)
    probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
    true_labels = predictions.label_ids
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = probs > threshold
        f1_macro = f1_score(true_labels, y_pred, average='macro', zero_division=0)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold
    
    return best_threshold, best_f1

def objective(trial):
    """Optuna objective function"""
    set_seed(42)
    
    train, val, test, emotion_cols, num_labels = load_data()
    
    model_name = "yandex/YandexGPT-5-Lite-8B-instruct"
    
    model, tokenizer = setup_model_and_tokenizer(model_name, num_labels, trial)
    model.print_trainable_parameters()
    
    train_ds, eval_ds, test_ds = tokenize_datasets(train, val, test, tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=16)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 4, 8])
    gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [2, 4, 8])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 7)
    
    training_args = TrainingArguments(
        output_dir='llm_cls_optuna',
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        group_by_length=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
    )
    
    # Trainer
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    
    best_threshold, best_f1_with_threshold = find_best_threshold(trainer, eval_ds)
    
    final_f1 = max(eval_results['eval_macro_f1'], best_f1_with_threshold)
    
    trial.report(final_f1, step=num_train_epochs)
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return final_f1

def evaluate_best_model(best_params):
    """Evaluate the best model on test set"""
    set_seed(42)
    
    train, val, test, emotion_cols, num_labels = load_data()
    
    model_name = "yandex/YandexGPT-5-Lite-8B-instruct"
    
    class MockTrial:
        def __init__(self, params):
            self.params = params
        
        def suggest_int(self, name, low, high, step=None):
            return self.params[name]
        
        def suggest_float(self, name, low, high, step=None, log=False):
            return self.params[name]
        
        def suggest_categorical(self, name, choices):
            return self.params[name]
    
    mock_trial = MockTrial(best_params)
    
    model, tokenizer = setup_model_and_tokenizer(model_name, num_labels, mock_trial)
    
    train_ds, eval_ds, test_ds = tokenize_datasets(train, val, test, tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=16)
    
    training_args = TrainingArguments(
        output_dir='llm_cls_optuna',
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        learning_rate=best_params['learning_rate'],
        num_train_epochs=best_params['num_train_epochs'],
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        group_by_length=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    print("Training best model...")
    trainer.train()
    
    print("Finding best threshold...")
    best_threshold, _ = find_best_threshold(trainer, eval_ds)
    print(f"Best threshold: {best_threshold:.2f}")
    
    print("Evaluating on test set...")
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_predictions = trainer.predict(test_ds)
    test_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).numpy()
    
    test_pred_labels = np.zeros_like(test_probs)
    for i, emotion in enumerate(emotion_cols):
        test_pred_labels[:, i] = (test_probs[:, i] > 0).astype(int)
    true_test_labels = test_predictions.label_ids
    
    print("\nTest Results:")
    for average in ['micro', 'macro']:
        recall = recall_score(true_test_labels, test_pred_labels, average=average, zero_division=0)
        precision = precision_score(true_test_labels, test_pred_labels, average=average, zero_division=0)
        f1 = f1_score(true_test_labels, test_pred_labels, average=average, zero_division=0)
        print(f'{average.upper()} recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}')

    print(f"\nPer-class Results:")
    class_recall = recall_score(true_test_labels, test_pred_labels, average=None, zero_division=0)
    class_precision = precision_score(true_test_labels, test_pred_labels, average=None, zero_division=0)
    class_f1 = f1_score(true_test_labels, test_pred_labels, average=None, zero_division=0)

    for i, emotion in enumerate(emotion_cols):
        print(f'{emotion.upper()}: recall: {class_recall[i]:.4f}, precision: {class_precision[i]:.4f}, f1: {class_f1[i]:.4f}')

    print(f"\nClass distribution in test set:")
    for i, emotion in enumerate(emotion_cols):
        true_count = int(true_test_labels[:, i].sum())
        pred_count = int(test_pred_labels[:, i].sum())
        total = len(true_test_labels)
        print(f'{emotion.upper()}: true: {true_count}/{total} ({true_count/total:.1%}), predicted: {pred_count}/{total} ({pred_count/total:.1%})')

def main():
    """Main function to run hyperparameter optimization"""
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=10)
    
    print("\nBest trial:")
    print(f"Value: {study.best_value:.4f}")
    print("Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*50)
    evaluate_best_model(study.best_params)

if __name__ == "__main__":
    main()