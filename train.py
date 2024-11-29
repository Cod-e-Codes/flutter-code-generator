# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# import torch

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Step 1: Load the dataset
# dataset = load_dataset("wraps/codegen-flutter-v1")

# # Step 2: Load the tokenizer and model
# model_name = "Salesforce/codegen-350M-mono"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# # Step 3: Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=512)

# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["content"])

# # Step 4: Set up training arguments
# training_args = TrainingArguments(
#     output_dir="./flutter_codegen_model",
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,  # Adjust based on GPU memory
#     num_train_epochs=3,
#     save_steps=500,
#     save_total_limit=2,
#     fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
#     logging_dir="./logs",
#     logging_steps=10,
#     report_to="none"
# )

# # Step 5: Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     tokenizer=tokenizer,
# )

# # Step 6: Train the model
# trainer.train()

# # Step 7: Save the fine-tuned model
# model.save_pretrained("./flutter_codegen_model")
# tokenizer.save_pretrained("./flutter_codegen_model")

# # # # # # # # # # # # # # # # #
#   Train on multiple datasets  #
# # # # # # # # # # # # # # # # #

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the datasets
print("Loading datasets...")
dataset1 = load_dataset("wraps/codegen-flutter-v1")
dataset2 = load_dataset("limcheekin/flutter-website-3.7")
dataset3 = load_dataset("deepklarity/top-flutter-packages")

# Step 2: Preprocess datasets to extract relevant text
def preprocess_dataset1(example):
    return {"text": example["content"]}

def preprocess_dataset2(example):
    return {"text": example["text"]}

def preprocess_dataset3(example):
    # Combine title and description into one text entry
    return {"text": f"{example['title']} - {example['description']}"}

print("Preprocessing datasets...")
dataset1_train = dataset1["train"].map(preprocess_dataset1, remove_columns=["repo_id", "file_path", "content", "__index_level_0__"])
dataset2_train = dataset2["train"].map(preprocess_dataset2, remove_columns=["id", "source"])
dataset3_train = dataset3["train"].map(preprocess_dataset3, remove_columns=["title", "description", "likes", "dependencies"])

# Combine all datasets into a single dataset
print("Combining datasets...")
combined_dataset = concatenate_datasets([dataset1_train, dataset2_train, dataset3_train])

# Step 3: Create train-validation split
print("Creating train-validation split...")
train_test_split = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
validation_dataset = train_test_split["test"]

# Step 4: Load the tokenizer and model from the checkpoint
print("Loading tokenizer and model from checkpoint...")
checkpoint_path = "./flutter_codegen_model/checkpoint-1500"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)

# Step 5: Tokenize the datasets
def tokenize_function(examples):
    # Tokenize the text and add labels
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # Duplicate input_ids as labels
    return tokenized

print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 6: Set up training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./flutter_codegen_model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
    logging_dir="./logs",
    logging_steps=10,
    resume_from_checkpoint=checkpoint_path,  # Resume from the checkpoint
    report_to="none"
)

# Step 7: Initialize the Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,  # Use the new validation dataset
    tokenizer=tokenizer,
)

# Step 8: Train the model
print("Starting training from checkpoint...")
trainer.train()

# Step 9: Save the fine-tuned model
print("Saving the model...")
model.save_pretrained("./flutter_codegen_model")
tokenizer.save_pretrained("./flutter_codegen_model")

print("Training complete. Model saved to './flutter_codegen_model'.")
