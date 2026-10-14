import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

FEW_SHOT_PROMPT = """--- Example 1 ---
Context: sent1: leo is a kind of constellation sent2: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year sent3: a constellation contains stars

Question: Stars are organized into patterns called constellations. One constellation is named Leo. Which statement best explains why Leo appears in different areas of the sky throughout the year?

Question: What can we deduce in one logical step?
Answer: Leo is a constellation containing stars.

Question: What can we deduce in one logical step?
Answer: The Earth revolving around the Sun causes Leo to appear in different areas in the sky at different times of year.

--- Example 2 ---
Context: sent1: diurnal motion is when objects in the sky appear to move due to earth 's rotation on its axis sent2: stars appear to move relative to the horizon during the night sent3: the earth rotating on its axis causes stars / the moon to appear to move across the sky at night

Question: Which of the following statements best explains why stars appear to move across Earth's sky each night?

Question: What can we deduce in one logical step?
Answer: Stars appearing to move relative to the horizon during the night is an example of diurnal motion.

Question: What can we deduce in one logical step?
Answer: The Earth rotating on its axis causes stars to appear to move relative to the horizon during the night.

--- Example 3 ---
Context: sent1: more light gets reflected on highly reflective things sent2: venus is covered in highly reflective clouds sent3: as the light reflected off of an object increases , the object will appear to be brighter

Question: Kevin is observing the sky on a clear night. With the unaided eye he is able to see Venus, Mars, Jupiter, and Saturn. Why would Venus appear to be brighter than the other planets?

Question: What can we deduce in one logical step?
Answer: Venus reflects more light toward Earth than other planets because it is covered in highly reflective clouds.

Question: What can we deduce in one logical step?
Answer: As Venus reflects more light, it appears brighter than other planets."""

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
        
    data_path = "data/entailmentbank/test_decomposed.json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    results = []
    
    print("Running EntailmentBank Few-Shot Iterative...")
    for example in tqdm(dataset[:100]): # Limit for reasonable eval time
        question = example["question"]
        context_string = example["context_string"]
        hypothesis = example["hypothesis"]
        
        # Base context with clean spacing
        context = f"{FEW_SHOT_PROMPT}\n\nContext: {context_string}\n\nQuestion: {question}"
        
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
            "id": example["id"],
            "hypothesis": hypothesis,
            "generated_steps": generated_steps
        })
        
    with open("predictions_entailment_1.5b_fewshot.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("EntailmentBank evaluation complete.")

if __name__ == "__main__":
    main()
