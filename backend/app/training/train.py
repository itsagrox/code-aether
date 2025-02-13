from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("csv", data_files={"train": "./train.csv"})

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    inputs = [
        (instr or "") + "\n" + (inp or "") 
        for instr, inp in zip(examples["instruction"], examples["input"])
    ]
    targets = [out for out in examples["output"]]

    model_inputs = tokenizer(
        inputs, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )

    labels = tokenizer(
        targets, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )["input_ids"]

    # Replace padding tokens in labels with -100 so they are ignored in loss computation
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]

    model_inputs["labels"] = labels
    return model_inputs


# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]

# Enable 4-bit quantization with bitsandbytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for stability
    bnb_4bit_use_double_quant=True  # Double quantization for efficiency
)

# Load quantized model with auto offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Offload to CPU when needed
    torch_dtype=torch.float16  # Force mixed precision
)

# Disable gradient checkpointing to prevent conflicts
model.gradient_checkpointing_disable()

# Apply LoRA fine-tuning (lower rank for lower memory usage)
lora_config = LoraConfig(
    r=2,  # Reduce LoRA rank
    lora_alpha=4,  # Lower alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Ensure use_cache=False to avoid memory issues
model.config.use_cache = False

# Define training arguments optimized for low VRAM
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Lower batch size
    gradient_accumulation_steps=16,  # Simulate a batch size of 16
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    report_to="none",
    fp16=True,  # Enable mixed precision training
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
