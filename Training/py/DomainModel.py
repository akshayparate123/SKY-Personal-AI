from datasets import load_dataset
import warnings
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")
import torch
torch.cuda.empty_cache()
from datetime import datetime
# Get the current date and time
now = datetime.now()
current_date_time = now.strftime("%Y_%m_%d")
import logging
from transformers import DataCollatorForSeq2Seq



#####################Training Parameters##############################

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
modelName = "Domain"
max_len = 64
# op_max_len = 64
batch_size = 100
epochs = 4
numberOfWorkers = 0

################L###ogging################################
# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler("../logs/{}_{}.log".format(modelName,current_date_time)),  # Log messages to a file
        logging.StreamHandler()  # Also print log messages to console
    ]
)

# Create a logger
logger = logging.getLogger(__name__)

# Log messages
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')


#####################Cuda Enable###########################
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#####################Dataset Loading##############################

dataset = pd.read_csv("../Data/CleanedDatasets/Domain.csv")
# dataset = load_dataset("Ateeqq/news-title-generator")
X_train, X_test, y_train, y_test = train_test_split(dataset["text"],dataset["Domain"], test_size=0.2, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size=0.5, random_state=42)
train_df = pd.DataFrame()
valid_df = pd.DataFrame()
train_df["text"] = X_train
train_df["Domain"] = y_train
valid_df["text"] = X_valid
valid_df["Domain"] = y_valid

#####################Loading Pre Trained Model##############################

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 
#####################Loading saved model ##############################

# model_name = '../Saved_Models/{}/fine-tuned-bert-sentiment_{}'.format(modelName,"02-07-2024_3")
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)
# model = model.to(device)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#####################Data Loader##############################

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

#####################Train Model##############################

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for idx,d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if idx % 1000 == 0 and idx != 0:
            logger.info("Batch Number : {}, Training Loss : {}".format(idx,np.mean(losses)))
            # print("Batch Number : {}, Training Loss : {}".format(idx,np.mean(losses)))
        elif idx % 1 == 0 and idx != 0:
            print(f'\rTraining Progress: {idx}/{len(data_loader)} Loss : {loss}', end='', flush=True)
    return correct_predictions.double() / n_examples, np.mean(losses)

#####################Validation Model##############################

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for idx,d in enumerate(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            if idx % 1000 == 0 and idx != 0:
                logger.info("Batch Number : {}, Validation Loss : {}".format(idx,np.mean(losses)))
                # print("Batch Number : {}, Validation Loss : {}".format(idx,np.mean(losses)))
                # print()
            elif idx % 1 == 0:
                print(f'\rValidation Progress: {idx}/{len(data_loader)}  Loss : {loss}', end='', flush=True)
    return correct_predictions.double() / n_examples, np.mean(losses)


#####################Create DataLoader################################

train_dataset = TextDataset(
    texts=train_df['text'].to_numpy(),
    labels=train_df['Domain'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len

)

val_dataset = TextDataset(
    texts=valid_df['text'].to_numpy(),
    labels=valid_df['Domain'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

#####################Training################################
def main():
    trainingLoss = []
    validationLoss = []
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=numberOfWorkers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=numberOfWorkers)

    optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
    total_steps = len(train_loader) * epochs  # 4 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1).to(device)
    for epoch in range(epochs):
        logger.info("Training Loss : {}".format(trainingLoss))
        logger.info("Validation Loss : {}".format(validationLoss))
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        correct_prediction,train_loss = train_model(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_dataset)
        )
        logger.info(f'Train loss {train_loss} & Correct Prediction {correct_prediction}')
        trainingLoss.append(train_loss)
        correct_prediction,val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(val_dataset)
        )
        logger.info(f'Validation loss {val_loss} & Correct Prediction {correct_prediction}')
        validationLoss.append(val_loss)
        if(len(validationLoss) > 1):
            if validationLoss[-1] > validationLoss[-2]:
                logger.critical('Validation loss increased.')
                logger.critical('Model is overfitting')
                logger.critical('Model Training Stopped')
                break
            else:
                pass
        else:
            pass
        model.save_pretrained('../Saved_Models/{}/fine-tuned-bert-sentiment_{}_{}'.format(modelName,current_date_time,epoch))
        tokenizer.save_pretrained('../Saved_Models/{}/fine-tuned-bert-sentiment_{}_{}'.format(modelName,current_date_time,epoch))
        logger.info("Model Saved")

# if __name__ == '__main__':
#     main()