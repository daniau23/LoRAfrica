import os
import torch
import numpy as np
import wandb
from datasets import load_dataset, Dataset
from transformers import pipeline
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from bert_score import score as bert_score
from huggingface_hub import login

HF_TOKEN = "your_token_here"  # Replace with your actual token
WANDB_KEY = "your_wandb_key_here"  # Replace with your actual WandB key

# Setup - Replace with your info
login(token=HF_TOKEN)

# Initialize W&B
# In a distributed setup, the Trainer handles the rank-check for you
# Only login/init wandb on the main process (Rank 0)
if int(os.environ.get("RANK", 0)) == 0:
    wandb.login(key=WANDB_KEY)
    wandb.init(project="phi4_african_history", name="eval_axolotl_phi4_african_history_lora")

model_id = "microsoft/Phi-4-mini-instruct"
# output_dir = "./phi4_african_history_lora_ds2"
repo_id = "DannyAI/phi4_lora_axolotl" # replace with your HF repo

# Load Data
full_dataset = load_dataset("DannyAI/African-History-QA-Dataset")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

test_data = full_dataset["test"]

tokeniser = AutoTokenizer.from_pretrained(model_id)
tokeniser.pad_token = tokeniser.eos_token

# load base model
model  = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    trust_remote_code=False
)

# load LoRA layers
lora_model = PeftModel.from_pretrained(
    model,output_dir
)

lora_model.eval()

# The pipeline handles chat templates and decoding automatically
generator = pipeline(
    "text-generation",
    model=lora_model,
    tokenizer=tokeniser,
)

def generate_answer(question):
    """
    Generates an answer for the given question using the fine-tuned LoRA model.
    """
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialised in African history which gives concise answers to questions asked."},
        {"role": "user", "content": question}
    ]
    
    # pipeline() returns a list of dicts; return_full_text=False gives only the assistant's reply
    output = generator(
        messages, 
        max_new_tokens=2048, 
        temperature=0.1, 
        do_sample=False,
        return_full_text=False
    )
    return output[0]['generated_text'].strip()

# Generate predictions on the test set
print("--- Generating Predictions on Test Set ---")
test_predictions = []
# Assuming test_data is a list of dicts with "question" and "answer" keys
test_references = [item["answer"] for item in test_data]

for i, item in enumerate(test_data):
    pred = generate_answer(item["question"])
    test_predictions.append(pred)
    
    if i < 2: # Sample output for verification
        print(f"\nSample Q: {item['question']}")
        print(f"Sample A (Lora Model): {pred}")
        print(f"Sample A (Ref): {item['answer']}\n")

# Metrics Calculation using BERTScore
print("--- Calculating BERTScore ---")
# P = Precision, R = Recall, F1 = F1 Score
P, R, F1 = bert_score(test_predictions, test_references, lang="en", verbose=True)

avg_f1 = F1.mean().item()
print(f"\nFinal Evaluation Results:")
print(f"Average BERTScore F1: {avg_f1:.4f}")

# Finalize W&B
wandb.log({"Final_Test_BERTScore": avg_f1})
wandb.finish()