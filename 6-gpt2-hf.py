# GPT-2 Text Generation with Hugging Face Transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Text generation function
def generate_text_gpt2(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time, a wise dragon"
print("\nGenerated Text:")
print(generate_text_gpt2(prompt))
