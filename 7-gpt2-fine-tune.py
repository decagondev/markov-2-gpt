# GPT-2 Fine-Tuning and Text Generation with Hugging Face Transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch
import os

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Optional: add pad token if missing
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Save sample training data
train_text = """
Once upon a time, in a land far away, there lived a wise dragon.
The dragon loved to read books and teach children about the stars.
Every evening, the dragon would gather all the young animals and tell them stories of ancient times.
"""

with open("train.txt", "w") as f:
    f.write(train_text)

# Load dataset for fine-tuning
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=64
    )

dataset = load_dataset("train.txt", tokenizer)

# Data collator for language modeling
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=1,
    logging_steps=5,
    logging_dir="logs",
    report_to="none"
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator
)

# Fine-tune GPT-2
trainer.train()

# Save fine-tuned model
model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")

# Text generation with fine-tuned model
def generate_text_gpt2(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Every evening, the dragon"
print("\nGenerated Text:")
print(generate_text_gpt2(prompt))
