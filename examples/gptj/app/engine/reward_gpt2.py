import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = None
tokenizer = None
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'


def load_gpt2_reward_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2',
        truncation_side='left',
        padding_side='right'
    )
    tokenizer.pad_token_id = 50256
    model = AutoModelForSequenceClassification.from_pretrained("ChaiML/gpt2_base_retry_and_continue_12m_reward_model")
    model = model.to(DEVICE).eval()
    logging.info("loaded gpt2_reward_model")


def score(text: str):
    try:
        tokens = tokenizer(
            text,
            return_tensors='pt',
            return_attention_mask=True,
            padding='longest',
            truncation=True,
            max_length=256
        ).to(DEVICE)
        logits = torch.softmax(model(**tokens).logits, dim=1)
        preds = float(logits[0][1])
        return preds
    except Exception as e:
        logging.exception(e)
        return []
