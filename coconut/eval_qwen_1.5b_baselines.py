import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Qwen2.5-1.5B-Coconut model in bfloat16...")
    
    model_id = "CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Load dataset
    data_path = "data/entailmentbank/test_decomposed.json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    results_a = []
    results_b = []
    
    print("Running baselines on EntailmentBank...")
    for example in tqdm(dataset[:100]): # Limit to 100 for reasonable eval time
        question = example["question"]
        context_string = example["context_string"]
        
        # Baseline A: Original Dataset (No context facts)
        prompt_a = f"Question: {question}\nAnswer:"
        inputs_a = tokenizer(prompt_a, return_tensors="pt").to(device)
        
        # Baseline B: All-at-once
        prompt_b = f"Context: {context_string}\nQuestion: {question}\nAnswer:"
        inputs_b = tokenizer(prompt_b, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_a = model.generate(
                **inputs_a,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            outputs_b = model.generate(
                **inputs_b,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        gen_a = tokenizer.decode(outputs_a[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True).strip()
        gen_b = tokenizer.decode(outputs_b[0][inputs_b.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        results_a.append({
            "id": example["id"],
            "prompt": prompt_a,
            "generated": gen_a,
            "gold": example["hypothesis"]
        })
        
        results_b.append({
            "id": example["id"],
            "prompt": prompt_b,
            "generated": gen_b,
            "gold": example["hypothesis"]
        })
        
    with open("predictions_qwen_1.5b_baseline_a.json", "w") as f:
        json.dump(results_a, f, indent=2)
        
    with open("predictions_qwen_1.5b_baseline_b.json", "w") as f:
        json.dump(results_b, f, indent=2)
        
    print("Baselines complete.")

if __name__ == "__main__":
    main()
