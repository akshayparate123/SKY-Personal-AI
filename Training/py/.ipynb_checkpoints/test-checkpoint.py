from datasets import load_dataset
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

dataset_2 = load_dataset("Ateeqq/news-title-generator")

X_train, X_test, y_train, y_test = train_test_split(dataset_2["train"]["text"],dataset_2["train"]["summary"], test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(dataset_1["text"],dataset_1["summary"], test_size=0.2, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size=0.5, random_state=42)

import torch
torch.cuda.empty_cache()
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
model = model.to(device)


from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, input_texts, output_texts, tokenizer, max_length):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        input_tokenized = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        output_tokenized = self.tokenizer(output_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        input_ids = input_tokenized['input_ids'].squeeze(0)
        attention_mask = input_tokenized['attention_mask'].squeeze(0)
        labels = output_tokenized['input_ids'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=128)
valdiation_dataset = TextDataset(X_valid, y_valid, tokenizer, max_length=128)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Increase the number of epochs
    per_device_train_batch_size=64,  # Adjust batch size as per your GPU memory
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,  # Gradient accumulation to simulate larger batch size#
    fp16=False,  # Enable mixed precision training#
    learning_rate=2e-5,  # Adjust learning rate#
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=False,
    weight_decay=0.01,  # Regularization
    warmup_steps=500,  # Learning rate warmup
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = valdiation_dataset,
    data_collator=data_collator,
)

model.save_pretrained('../Saved_Models/TitleExtraction/fine-tuned-bert-sentiment_{}'.format("30-06-2024_"))
tokenizer.save_pretrained('../Saved_Models/TitleExtraction/fine-tuned-bert-sentiment_{}'.format("30-06-2024_"))
trainer.save_state()
# Train the model
trainer.train()