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
from datetime import datetime
# Get the current date and time
now = datetime.now()
current_date_time = now.strftime("%Y_%m_%d")
import logging
from rouge_score import rouge_scorer
#####################Training Parameters##############################

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
modelName = "RAGAgent"
ip_max_len = 1024
op_max_len = 100
batch_size = 15
epochs = 1
numberOfWorkers = 0
load_checkpoint = False
model_name = '../Saved_Models/{}/fine-tuned-bert-sentiment_{}'.format(modelName,"2024_12_02_0")
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

training_dataset = pd.read_csv("../Data/CleanedDatasets/{}_{}.tsv".format("RAGAgent","train"),sep='\t')[:1000000]
testing_dataset = pd.read_csv("../Data/CleanedDatasets/{}_{}.tsv".format("RAGAgent","test"),sep='\t')[:100000]

X_test, X_valid, y_test, y_valid = train_test_split(testing_dataset["network"],testing_dataset["path"], test_size=0.5, random_state=42)

train_df = pd.DataFrame()
valid_df = pd.DataFrame()
train_df["agent_1"] = training_dataset["network"]
train_df["agent_2"] = training_dataset["path"]
valid_df["agent_1"] = X_valid
valid_df["agent_2"] = y_valid

####################Loading Pre Trained Model##############################

# model_name = "BART"
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
# model = model.to(device)

# ####################Loading saved model ##############################

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# #####################Rouge Score Calculations##############################
# Initialize the ROUGE scorer
def rouge_calculate(reference,candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure,scores['rougeL'].fmeasure

#####################Data Loader##############################
logger.info("Model Name: {}".format(model_name))
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,ip_max_len,op_max_len):
        self.input_texts = texts
        self.output_texts = labels
        self.tokenizer = tokenizer
        self.ip_max_len = ip_max_len
        self.op_max_len = op_max_len
        

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        input_tokenized = self.tokenizer(input_text, max_length=self.ip_max_len, truncation=True, padding='max_length', return_tensors='pt',return_attention_mask=True)
        output_tokenized = self.tokenizer(output_text, max_length=self.op_max_len, truncation=True, padding='max_length', return_tensors='pt')

        input_ids = input_tokenized['input_ids'].squeeze()
        attention_mask = input_tokenized['attention_mask'].squeeze()
        output_ids = output_tokenized['input_ids'].squeeze()
        # output_ids[output_ids == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_ids': output_ids
        }

#####################Train Model##############################

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples,epoch):
    model.train()
    losses = []
    try:
        for idx,d in enumerate(data_loader):
            # print(f'\rTraining Progress: {idx}/{len(data_loader)}', end='', flush=True)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["output_ids"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = labels
            )        
            logits = outputs.logits.transpose(1,2)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if idx % 10000 == 0 and idx != 0:
                logger.info("Saving checkpoint")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
                torch.save(checkpoint, '../Checkpoints/checkpoint_{}.pth'.format(idx))
            elif idx % 1000 == 0 and idx != 0:
                logger.info("Batch Number : {}, Training Loss : {}".format(idx,np.mean(losses)))
                # print("Batch Number : {}, Training Loss : {}".format(idx,np.mean(losses)))

            elif idx % 1 == 0 and idx != 0:
                print(f'\rTraining Progress: {idx}/{len(data_loader)} Loss : {loss}', end='', flush=True)
            # print()
    except Exception as e:
        logger.error(e) 
    return np.mean(losses)

#####################Validation Model##############################

def eval_model(model, data_loader, loss_fn, device, n_examples,epoch):
    model.eval()
    losses = []
    with torch.no_grad(): 
        try:
            for idx,d in enumerate(data_loader):
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["output_ids"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels = labels
                )
                # loss = outputs.loss
                logits = outputs.logits.transpose(1,2)
                # _, preds = torch.max(logits, dim=1)
                loss = loss_fn(logits, labels)
                losses.append(loss.item())
                if idx % 1000 == 0 and idx != 0:
                    logger.info("Batch Number : {}, Validation Loss : {}".format(idx,np.mean(losses)))
                    # print("Batch Number : {}, Validation Loss : {}".format(idx,np.mean(losses)))
                    # print()
                elif idx % 1 == 0:
                    print(f'\rValidation Progress: {idx}/{len(data_loader)}  Loss : {loss}', end='', flush=True)
        except Exception as e:
            logger.error(e)
    return np.mean(losses)


#####################Test Model##############################
def test_model(model,tokenizer,X_test,y_test):
    rogue_score_1 = []
    rogue_score_L = []
    for counter in range(0, 1000, 10):
        try:
            print(f'\rProgress: {counter}/{1000}', end='', flush=True)
            batch = X_test.tolist()[counter:counter + batch_size]
            tokenized_batch = tokenizer(
                batch,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)  # Ensure tokenized inputs are on the same device as the model
            outputs = model.generate(
                input_ids=tokenized_batch["input_ids"],
                attention_mask=tokenized_batch["attention_mask"],
                max_length=100,
                num_beams=1,
                early_stopping=True
            )
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for idx,op in enumerate(decoded_outputs):
                r1,r2 = rouge_calculate(y_test.tolist()[idx],op)
                rogue_score_1.append(r1)
                rogue_score_L.append(r2)
        except Exception as e:
            logger.error(e)
    return np.mean(rogue_score_1),np.mean(rogue_score_L)
#####################Create DataLoader################################

train_dataset = TextDataset(
    texts=train_df['agent_1'].to_numpy(),
    labels=train_df['agent_2'].to_numpy(),
    tokenizer=tokenizer,
    ip_max_len = ip_max_len,
    op_max_len = op_max_len
)

val_dataset = TextDataset(
    texts=valid_df['agent_1'].to_numpy(),
    labels=valid_df['agent_2'].to_numpy(),
    tokenizer=tokenizer,
    ip_max_len = ip_max_len,
    op_max_len = op_max_len
)

#####################Training################################
def main():
    trainingLoss = []
    validationLoss = []
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=numberOfWorkers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=numberOfWorkers)
    optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
    
    if load_checkpoint:
        logger.info("Checkpoint loaded")
        checkpoint = torch.load('../Checkpoints/checkpoint_20000.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        train_loss = train_model(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_dataset),
            epoch
        )
        logger.info(f'Train loss {train_loss}')
        trainingLoss.append(train_loss)
        val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(val_dataset),
            epoch
        )
        logger.info(f'Validation loss {val_loss}')
        validationLoss.append(val_loss)
        now = datetime.now()
        current_date_time = now.strftime("%Y_%m_%d")
        model.save_pretrained('../Saved_Models/{}/fine-tuned-bert-sentiment_{}_{}'.format(modelName,current_date_time,epoch))
        tokenizer.save_pretrained('../Saved_Models/{}/fine-tuned-bert-sentiment_{}_{}'.format(modelName,current_date_time,epoch))
        logger.info("Model Saved")
        logger.info("Calculating Rouge Score of the model...")
        # r1,r2 = test_model(model,tokenizer,X_test,y_test)
        logger.info("Model Testing Complete\n1)rogue_score_1:{}\n2)rogue_score_L:{}".format(r1,r2))
if __name__ == '__main__':
    main()