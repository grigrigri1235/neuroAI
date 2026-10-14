import os
import sys
import logging
import json
import re

# Suppress TF warnings BEFORE importing bleurt/tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from bleurt import score
logging.getLogger("tensorflow").setLevel(logging.FATAL)

def extract_answer(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = re.sub(r'[^\d\.-]', '', ans)
        return ans
    return None

def extract_last_number(text):
    """Extract the last number appearing in the text as the model's answer."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        last = numbers[-1]
        if '.' in last:
            last = last.rstrip('0').rstrip('.')
        return last
    return None

def score_gsm8k(filename, label):
    if not os.path.exists(filename):
        print(f"Skipping {label}: {filename} not found", flush=True)
        return
        
    with open(filename, 'r') as f:
        data = json.load(f)
        
    correct = 0
    total = len(data)
    if total == 0:
        return
        
    for ex in data:
        gold_ans = extract_answer(ex['gold_answer'])
        last_step = ex['generated_steps'][-1] if ex['generated_steps'] else ""
        gen_ans = extract_last_number(last_step)
                
        if gold_ans is not None and gen_ans is not None and gold_ans == gen_ans:
            correct += 1
            
    print(f"GSM8K [{label}] Accuracy: {correct}/{total} ({correct/total*100:.2f}%)", flush=True)

def score_entailment(scorer, filename, label):
    if not os.path.exists(filename):
        print(f"Skipping {label}: {filename} not found", flush=True)
        return
        
    with open(filename, 'r') as f:
        data = json.load(f)
        
    candidates = []
    references = []
    
    for ex in data:
        final_step = ex['generated_steps'][-1] if ex['generated_steps'] else ""
        for step in ex['generated_steps']:
            if "####" in step:
                final_step = step.replace("####", "").strip()
                break
                
        candidates.append(final_step)
        references.append(ex['hypothesis'])
        
    scores = scorer.score(references=references, candidates=candidates)
    avg_score = sum(scores) / len(scores)
    
    print(f"EntailmentBank [{label}] Avg BLEURT: {avg_score:.4f}", flush=True)

if __name__ == "__main__":
    print("=== Evaluating Results ===", flush=True)
    
    # 1. GSM8K Evaluations
    score_gsm8k('predictions_gsm8k_1.5b_less_handholding.json', 'Iterative Latents')
    score_gsm8k('predictions_gsm8k_1.5b_no_latents.json', 'No-Latents Control')
    
    # 2. EntailmentBank Evaluations (Load BLEURT once to save memory & time)
    print("Loading BLEURT-Large-512...", flush=True)
    bleurt_scorer = score.BleurtScorer("bleurt-large-512")
    score_entailment(bleurt_scorer, 'predictions_entailment_1.5b_fewshot.json', 'Iterative Latents')
    score_entailment(bleurt_scorer, 'predictions_entailment_1.5b_no_latents.json', 'No-Latents Control')
    
    print("=== Scoring Complete ===", flush=True)
