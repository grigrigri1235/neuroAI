import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

FEW_SHOT_PROMPT = """--- Example 1 ---
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Question: What can we deduce in one logical step?
Answer: Since there are 60 minutes in an hour, Weng earns $12 / 60 = $0.20 per minute.

Question: What can we deduce in one logical step?
Answer: Since she babysat for 50 minutes at $0.20 per minute, she earned 50 * 0.20 = $10. So she earned $10.

--- Example 2 ---
Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

Question: What can we deduce in one logical step?
Answer: Today Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today, for a total of 12 + 24 = 36 pages read so far.

Question: What can we deduce in one logical step?
Answer: There are 120 - 36 = 84 pages left. Half of the remaining pages is 84 / 2 = 42 pages. So she should read 42 pages tomorrow.

--- Example 3 ---
Question: Hans booked a room in a hotel. The hotel has 10 floors with 10 identical rooms on each floor. Because of an accident, the last floor is unavailable for the guests. Considering there are no other guests, in how many different rooms could Hans be checked in?

Question: What can we deduce in one logical step?
Answer: With 10 floors and 10 rooms per floor, the hotel has a total of 10 * 10 = 100 rooms.

Question: What can we deduce in one logical step?
Answer: The last floor of 10 rooms is unavailable, so there are 100 - 10 = 90 available rooms. So Hans could be checked into 90 different rooms."""

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
    
    # Ensure <c_thought> is in tokenizer
    if tokenizer.convert_tokens_to_ids("<c_thought>") in (None, tokenizer.unk_token_id):
        tokenizer.add_tokens(["<c_thought>"])
        model.resize_token_embeddings(len(tokenizer))
        print("Added <c_thought> to tokenizer and resized embeddings.")
        
    c_thought_id = tokenizer.convert_tokens_to_ids("<c_thought>")
    
    data_path = "data/gsm8k_test.json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    results = []
    
    print("Running GSM8K Less Handholding...")
    for example in tqdm(dataset[:100]): # Limit for reasonable eval time
        question = example["question"]
        gold_answer = example["answer"]
        
        # We start with the base context with clean spacing
        context = f"{FEW_SHOT_PROMPT}\n\nQuestion: {question}"
        
        generated_steps = []
        
        for stage in range(2): # Max 2 stages
            prompt = f"{context}\n\nQuestion: What can we deduce in one logical step?\n<c_thought><c_thought>\nAnswer: "
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            
            # Scrub hallucinated <c_thought> tokens if they appear in the generation
            gen_text = gen_text.replace("<c_thought>", "").replace("<|endoftext|>", "").strip()
            
            generated_steps.append(gen_text)
            
            # Append the cleaned text answer to the context
            context = f"{context}\nAnswer: {gen_text}"
            
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "generated_steps": generated_steps
        })
        
    with open("predictions_gsm8k_1.5b_less_handholding.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("GSM8K evaluation complete.")

if __name__ == "__main__":
    main()
