def generate_response(input_text,tokenizer,device,model):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=512, 
        num_beams=5, 
        early_stopping=True, 
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        num_return_sequences=1,  # Number of sequences to return
        temperature=0.7,  # Sampling temperature
        top_k=50,  # Top-K sampling
        top_p=0.9  # Top-p (nucleus) sampling
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response