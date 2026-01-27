import os
import torch
import wandb  # Added missing import
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
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
    wandb.init(project="phi4_african_history", name="phi4_african_history_lora_ds2")

model_id = "microsoft/Phi-4-mini-instruct" # Base model
output_dir = "./phi4_african_history_lora_ds2" # Output directory
repo_id = "DannyAI/phi4_african_history_lora_ds2" # Your HF repo

# Load Data
full_dataset = load_dataset("DannyAI/African-History-QA-Dataset")

# Load Tokenizer
tokeniser = AutoTokenizer.from_pretrained(model_id)
tokeniser.pad_token = tokeniser.eos_token

# Load Model (No device_map for DeepSpeed!)
# Base model loaded in bfloat16 for DeepSpeed Stage 2
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False
)
# LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA model
model = get_peft_model(model, lora_config)

# Tokenization & Collator
def tokenisation(example)->dict:
    """Tokenizes the input example by applying the chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialised in African history."},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    full_text = tokeniser.apply_chat_template(messages, tokenize=False)
    return tokeniser(full_text, truncation=True, max_length=2048, add_special_tokens=False)

train_dataset = full_dataset["train"].map(tokenisation)
val_dataset = full_dataset["validation"].map(tokenisation)

# Custom data collator to mask inputs before assistant response
class AssistantMaskingCollator():
    def __init__(self, tokeniser):
        self.tokeniser = tokeniser
        self.assistant_header = tokeniser.encode("<|assistant|>\n", add_special_tokens=False)

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokeniser.pad_token_id)
        labels = input_ids.clone()
        for i in range(len(labels)):
            found = False
            for j in range(len(input_ids[i]) - len(self.assistant_header) + 1):
                if input_ids[i][j : j + len(self.assistant_header)].tolist() == self.assistant_header:
                    labels[i, : j + len(self.assistant_header)] = -100
                    found = True
                    break
            labels[i][input_ids[i] == self.tokeniser.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(self.tokeniser.pad_token_id).long(), "labels": labels}

data_collator= AssistantMaskingCollator(tokeniser)

# Training
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2, # Adjusted for DeepSpeed
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=10,
    bf16=True,
    eval_strategy="steps",             
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    report_to="wandb",
    remove_unused_columns=False,       
    deepspeed="ds_config_2.json", # DeepSpeed config file
    push_to_hub=True,
    hub_model_id=repo_id
)

# Initialize Trainer with training arguments, datasets, model, and data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Log metrics to W&B
# This saves the LoRA adapter and the tokenizer
trainer.push_to_hub()
if int(os.environ.get("RANK", 0)) == 0:
    wandb.finish()

print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")