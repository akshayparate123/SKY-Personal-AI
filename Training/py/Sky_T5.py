from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
# tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16)