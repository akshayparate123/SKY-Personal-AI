import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AdamW
from datetime import datetime
now = datetime.now()
current_date_time = now.strftime("%Y_%m_%d")
modelName = "Sky"
model_name = '../Saved_Models/{}/fine-tuned-bert-sentiment_{}'.format(modelName,"2024_10_03_0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('../Checkpoints/checkpoint_40000.pth')
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.save_pretrained('../Saved_Models/{}/fine-tuned-bert-sentiment_{}_{}'.format(modelName,current_date_time,0))
tokenizer.save_pretrained('../Saved_Models/{}/fine-tuned-bert-sentiment_{}_{}'.format(modelName,current_date_time,0))