import os
import sys
import logging
import json
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from bleurt import score
logging.getLogger("tensorflow").setLevel(logging.FATAL)

def extract_last_number(text):
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        last = numbers[-1]
        if '.' in last:
            last = last.rstrip('0').rstrip('.')
        return last
    return None

def extract_answer(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = re.sub(r'[^\d\.-]', '', ans)
        return ans
    return None

def score_gsm8k(filename):
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        return
        
    with open(filename, 'r') as f:
        data = json.load(f)
        
    correct = 0
    total = len(data)
    if total == 0:
        return
        
    for ex in data:
        gold_ans = extract_answer(ex['gold_answer'])
        gen_ans = extract_last_number(ex['generated'])
        
        if gold_ans is not None and gen_ans is not None and gold_ans == gen_ans:
            correct += 1
            
    print(f"{filename} Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")

def score_entailment(filename):
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        return
        
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # Lazy load bleurt only if file exists to save memory
    bleurt_scorer = score.BleurtScorer("bleurt-large-512")
    
    candidates = []
    references = []
    
    for ex in data:
        candidates.append(ex['generated'])
        # The entailment baseline script saves the truth as 'gold'
        references.append(ex['gold'])
        
    scores = bleurt_scorer.score(references=references, candidates=candidates)
    avg_score = sum(scores) / len(scores)
    
    print(f"{filename} Avg BLEURT: {avg_score:.4f}")

if __name__ == "__main__":
    print("Evaluating Baseline Results...")
    
    score_gsm8k('predictions_gsm8k_1.5b_baseline_a.json')
    score_gsm8k('predictions_gsm8k_1.5b_baseline_b.json')
    
    score_entailment('predictions_qwen_1.5b_baseline_a.json')
    score_entailment('predictions_qwen_1.5b_baseline_b.json')
