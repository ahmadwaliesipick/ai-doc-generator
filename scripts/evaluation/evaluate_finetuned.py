from transformers import RobertaTokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from tqdm import tqdm
import evaluate
import os

def run_evaluation(test_csv, model_path="models/codet5_finetuned", output_path="metrics/finetuned_results.csv"):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please upload the fine-tuned model from Colab.")
        return

    print(f"Loading fine-tuned model from {model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Load test set
    df = pd.read_csv(test_csv)
    print(f"Evaluating fine-tuned model on {len(df)} samples...")
    
    results = []
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_text = f"summarize: {str(row['code'])}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=128)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            'code': row['code'],
            'actual': row['comment'],
            'predicted': prediction
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_df.to_csv(output_path, index=False)
    
    # Calculate scores
    rouge_score = rouge.compute(predictions=res_df['predicted'], references=res_df['actual'])
    bleu_score = bleu.compute(predictions=res_df['predicted'], references=res_df['actual'])
    
    print("\nFine-tuned Model Evaluation Complete:")
    print(f"ROUGE: {rouge_score}")
    print(f"BLEU: {bleu_score}")
    
    summary_path = "metrics/finetuned_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Fine-tuned Model Summary (CodeT5 small)\n")
        f.write(f"ROUGE: {rouge_score}\n")
        f.write(f"BLEU: {bleu_score}\n")
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    test_path = "data/processed/splits/test.csv"
    run_evaluation(test_path)
