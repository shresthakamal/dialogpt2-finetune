import torch
from utils import logging


def generate_inputs(utterance, history, tokenizer, effective_max_len=32):
    logger = logging()
    logger.info(f"Creating TensorDataset and DataLoader ...")

    input_ids = []
    attention_masks = []
    labels = []
    for i in range(len(utterance)):

        encoded_utterance = tokenizer.encode_plus(
            utterance[i].lower() + tokenizer.eos_token,
            max_length=effective_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoded_history = tokenizer.encode_plus(
            history[i].lower(), max_length=effective_max_len, truncation=True, padding="max_length", return_tensors="pt"
        )

        ids = torch.cat([encoded_utterance["input_ids"][0], encoded_history["input_ids"][0]], dim=0).reshape(
            1, effective_max_len * 2
        )
        mask = torch.cat([encoded_utterance["attention_mask"][0], encoded_history["attention_mask"][0]], dim=0).reshape(
            1, effective_max_len * 2
        )

        _label = torch.tensor([element if element != 50256 else -100 for element in encoded_history["input_ids"][0]])
        label = torch.cat([torch.full((effective_max_len,), -100), _label], dim=0).reshape(1, effective_max_len * 2)

        input_ids.append(ids)
        attention_masks.append(mask)
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    return input_ids, attention_masks, labels
