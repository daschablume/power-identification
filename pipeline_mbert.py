from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, concatenate_datasets, DatasetDict

# Load the datasets from two TSV files
dataset1 = load_dataset('csv', data_files='./data/orientation/orientation-ee-train.tsv', delimiter='\t')
dataset2 = load_dataset('csv', data_files='./data/orientation/orientation-fi-train.tsv', delimiter='\t')

# Combine the datasets
combined_dataset = concatenate_datasets([dataset1['train'], dataset2['train']])

# Split the combined dataset into train and test sets (e.g., 80-20 split)
split_combined_dataset = combined_dataset.train_test_split(test_size=0.2)

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=2)

# Tokenize the data
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_datasets = split_combined_dataset.map(preprocess_function, batched=True)

# Load evaluation metric
f1_metric = load_metric('f1')

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Smaller batch size
    per_device_eval_batch_size=4,   # Smaller batch size
    gradient_accumulation_steps=4,  # Accumulate gradients
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=lambda p: f1_metric.compute(predictions=p.predictions.argmax(-1), references=p.label_ids)
)

# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Make predictions on the test set
predictions = trainer.predict(tokenized_datasets['test'])
print(predictions.predictions)
