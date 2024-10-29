def get_chat_template(prompt, tokenizer):
    prompt = [
        {"role": "user", "content": prompt},
        ]
    prompt_with_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    return prompt_with_template