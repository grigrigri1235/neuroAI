import json
import os

data_dir = "/home/eran.b/neuroAI/coconut/data/entailmentbank"
splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]

for split in splits:
    in_path = os.path.join(data_dir, split)
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        continue
        
    out_name = split.replace(".jsonl", "_decomposed.json")
    out_path = os.path.join(data_dir, out_name)
    
    parsed_data = []
    
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            proof_str = row.get("proof", "")
            
            # Split by ';' and filter empty
            raw_steps = [s.strip() for s in proof_str.split(";") if s.strip()]
            
            steps = []
            for i, step in enumerate(raw_steps):
                if "->" not in step:
                    continue
                    
                left, right = step.split("->")
                left = left.strip()
                right = right.strip()
                
                # Get inputs
                inputs = [inp.strip() for inp in left.split("&")]
                
                # Format prompt
                if len(inputs) == 1:
                    prompt = f"What can be concluded from {inputs[0]}?"
                elif len(inputs) == 2:
                    prompt = f"What can be concluded from {inputs[0]} and {inputs[1]}?"
                else:
                    prompt = f"What can be concluded from {', '.join(inputs[:-1])} and {inputs[-1]}?"
                
                is_final = False
                target_text = ""
                
                if right == "hypothesis":
                    is_final = True
                    target_text = row.get("hypothesis", "")
                else:
                    # e.g., "int1: some text"
                    parts = right.split(":", 1)
                    if len(parts) == 2:
                        target_text = parts[1].strip()
                    else:
                        target_text = right.strip()
                
                steps.append({
                    "inputs": inputs,
                    "prompt": prompt,
                    "target_text": target_text,
                    "is_final": is_final
                })
                
            parsed_data.append({
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "hypothesis": row.get("hypothesis", ""),
                "context_string": row.get("context", ""),
                "steps": steps
            })
            
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2)
        
    print(f"Parsed {len(parsed_data)} items from {split} to {out_name}")

print("Done parsing.")
