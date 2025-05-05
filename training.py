from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
from datasets import load_dataset

# 1. Load custom dataset
dataset = load_dataset("json", data_files="Test13.jsonl", split="train")

# 2. Remove unused columns from the dataset
dataset = dataset.remove_columns(['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'])

# 3. Format the dataset examples
def format_example(example):
    return {
        "question": example["question"],
        "response": example["response"],
        "text": f"### Question:\n{example['question']}\n\n### Answer:\n{example['response']}"
    }

dataset = dataset.map(format_example)

# 4. Load the pre-trained model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set correctly
model = AutoModelForCausalLM.from_pretrained(model_name)

# 5. Apply LoRA configuration for fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, peft_config)

# 6. Tokenize the dataset with label masking for causal language modeling
def tokenize_function(example):
    prompt = f"### Question:\n{example['question']}\n\n### Answer:\n"
    full_text = example["text"]

    full_tokens = tokenizer(full_text, padding="max_length", truncation=True, max_length=512)
    prompt_tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

    labels = full_tokens["input_ids"].copy()
    prompt_len = sum([1 for token in prompt_tokens["input_ids"] if token != tokenizer.pad_token_id])
    labels[:prompt_len] = [-100] * prompt_len  # Mask the question for loss calculation

    full_tokens["labels"] = labels
    return full_tokens

tokenized_dataset = dataset.map(tokenize_function, remove_columns=["question", "response", "text"])

# 7. Training arguments for the Trainer class
training_args = TrainingArguments(
    output_dir="./results_lora",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# 8. Initialize the Trainer with model, training arguments, and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset  # For quick testing, using the same dataset for evaluation
)

# 9. Train the model
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./results_lora")
    tokenizer.save_pretrained("./results_lora")

    # 10. Load the fine-tuned model and tokenizer for inference
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    model.load_adapter("./results_lora", adapter_name="lora")
    tokenizer = AutoTokenizer.from_pretrained("./results_lora")
    tokenizer.pad_token = tokenizer.eos_token

    # 11. Generate text using the fine-tuned model
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    custom_question = "What are the services provided by the CloudTern Solutions?"
    prompt = f"### Question:\n{custom_question}\n\n### Answer:\n"
    output = generator(prompt, max_length=256, do_sample=True, top_p=0.9, temperature=0.7)
    print("\nGenerated Answer:\n", output[0]["generated_text"])
