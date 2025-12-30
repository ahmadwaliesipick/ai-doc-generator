from transformers import RobertaTokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from tqdm import tqdm
import evaluate
import os

def run_baseline(test_csv, model_name="Salesforce/codet5-small", output_path="metrics/baseline_results.csv"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    df = pd.read_csv(test_csv).head(100) # Small sample for baseline speed
    print(f"Evaluating baseline on {len(df)} samples...")
    
    results = []
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_text = f"summarize: {row['code']}"
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
    
    print("\nBaseline Evaluation Complete:")
    print(f"ROUGE: {rouge_score}")
    print(f"BLEU: {bleu_score}")
    
    with open("metrics/baseline_summary.txt", "w") as f:
        f.write(f"Baseline Summary (Zero-shot CodeT5)\n")
        f.write(f"ROUGE: {rouge_score}\n")
        f.write(f"BLEU: {bleu_score}\n")

if __name__ == "__main__":
    run_baseline("data/processed/splits/test.csv")
    print("Baseline evaluation results saved to metrics/baseline_summary.txt")
