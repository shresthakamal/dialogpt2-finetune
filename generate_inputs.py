import torch
from utils import logging
from make_dataset import prepare_data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler


def generate_inputs(file_path, tokenizer, effective_max_len=32, num_context=3, batch_size=4, verbose=False):
    logger = logging()

    utterance, history = prepare_data(
        file_path=file_path,
        tokenizer=tokenizer,
        num_context=num_context,
    )

    logger.info(f"Creating TensorDataset and DataLoader ...")

    input_ids = []
    attention_masks = []
    labels = []
    for i in range(len(utterance)):
        if verbose:
            i = 13
            print(utterance[i])
            print(history[i])

        encoded_utterance = tokenizer.encode_plus(
            utterance[i].lower() + tokenizer.eos_token,
            max_length=effective_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoded_history = tokenizer.encode_plus(
            history[i].lower(),
            max_length=effective_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        if verbose:
            print(encoded_utterance)
            print(encoded_history)

        ids = torch.cat([encoded_history["input_ids"][0], encoded_utterance["input_ids"][0]], dim=0).reshape(
            1, effective_max_len * 2
        )
        mask = torch.cat(
            [encoded_history["attention_mask"][0], encoded_utterance["attention_mask"][0]], dim=0
        ).reshape(1, effective_max_len * 2)

        _label = torch.tensor(
            [element if element != 50256 else -100 for element in encoded_utterance["input_ids"][0]]
        )

        label = torch.cat([torch.full((effective_max_len,), -100), _label], dim=0).reshape(
            1, effective_max_len * 2
        )

        if verbose:
            print(ids)
            print(mask)
            print(label)

        input_ids.append(ids)
        attention_masks.append(mask)
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

    return dataloader
