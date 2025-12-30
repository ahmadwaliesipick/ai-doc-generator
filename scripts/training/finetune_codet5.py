import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import os

def finetune_codet5(train_csv, val_csv, model_name="Salesforce/codet5-small", output_dir="models/codet5_finetuned"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    def preprocess_function(examples):
        inputs = ["summarize: " + code for code in examples["code"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        
        # Tokenize targets
        labels = tokenizer(text_target=examples["comment"], max_length=128, truncation=True, padding="max_length")
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    # Metrics
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use FP16 if GPU available
        logging_steps=100,
        push_to_hub=False,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import torch
    train_path = "data/processed/splits/train.csv"
    val_path = "data/processed/splits/val.csv"
    
    # Check for GPU and memory constraints
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # On CPU, we use a smaller subset to ensure the process completes in a reasonable time
    # This can be adjusted for a full run if needed
    train_df = pd.read_csv(train_path)
    if device == "cpu":
        print("CPU detected. Using a subset of 1000 samples for initial fine-tuning...")
        subset_train_path = "data/processed/splits/train_subset.csv"
        train_df.head(1000).to_csv(subset_train_path, index=False)
        train_path = subset_train_path
        
    finetune_codet5(train_path, val_path)
