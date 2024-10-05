from datasets import load_dataset
import warnings
import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")
import torch
torch.cuda.empty_cache()

#####################Cuda Enable###########################
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#####################Dataset Loading##############################

dataset_3 = load_dataset("dreamproit/bill_summary_us")
X_train, X_test, y_train, y_test = train_test_split(dataset_3["train"]["text"],dataset_3["train"]["title"], test_size=0.2, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size=0.5, random_state=42)
train_df = pd.DataFrame()
valid_df = pd.DataFrame()
train_df["text"] = X_train
train_df["title"] = y_train
valid_df["text"] = X_valid
valid_df["title"] = y_valid

#####################Loading Pre Trained Model##############################

# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
# model = model.to(device)

#####################Loading saved model ##############################

model_name = '../Saved_Models/TitleExtraction/fine-tuned-bert-sentiment_{}'.format("30-06-2024")
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#####################Data Loader##############################

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.input_texts = texts
        self.output_texts = labels
        self.tokenizer = tokenizer
        self.max_length = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        input_tokenized = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        output_tokenized = self.tokenizer(output_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        input_ids = input_tokenized['input_ids'].squeeze(0)
        attention_mask = input_tokenized['attention_mask'].squeeze(0)
        output_ids = output_tokenized['input_ids'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_ids': output_ids
        }

#####################Train Model##############################

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    for idx,d in enumerate(data_loader):
        # print(f'\rTraining Progress: {idx}/{len(data_loader)}', end='', flush=True)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        output_ids = d["output_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        targets = output_ids.view(-1)
        _, preds = torch.max(logits, dim=1)
        loss = loss_fn(logits, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if idx % 1000 == 0:
            print("Batch Number : {}, Training Loss : {}".format(idx,np.mean(losses)))
            print()
        elif idx % 1 == 0:
            print(f'\rTraining Progress: {idx}/{len(data_loader)}', end='', flush=True)
    return np.mean(losses)

#####################Validation Model##############################

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for idx,d in enumerate(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            output_ids = d["output_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits.view(-1, outputs.logits.size(-1))
            targets = output_ids.view(-1)
            _, preds = torch.max(logits, dim=1)
            loss = loss_fn(outputs.logits, output_ids)
            losses.append(loss.item())
            if idx % 1000 == 0:
                print("Batch Number : {}, Validation Loss : {}".format(idx,np.mean(losses)))
                print()
            elif idx % 1 == 0:
                print(f'\rValidation Progress: {idx}/{len(data_loader)}', end='', flush=True)
    return np.mean(losses)

#####################Training Parameters##############################

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  #This value determines the number of tokens to be considered for each text input.
batch_size = 16
epochs = 4

#####################Create DataLoader################################

train_dataset = TextDataset(
    texts=train_df['text'].to_numpy(),
    labels=train_df['title'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

val_dataset = TextDataset(
    texts=valid_df['text'].to_numpy(),
    labels=valid_df['title'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

#####################Training################################
def main():
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=3)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * 4  # 4 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        train_acc, train_loss = train_model(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_dataset)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(val_dataset)
        )
        print(f'Validation loss {val_loss} accuracy {val_acc}')


if __name__ == '__main__':
    main()