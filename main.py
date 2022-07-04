import os
import torch
import random
import argparse
import pickle
import pandas as pd
from pathlib import Path

from tqdm.auto import tqdm
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


def seputterances(row):
    try:
        row = row.split("__eou__")
        row = row[:-1]
        return row
    except:
        return row


def evaluate(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Working on GPU: {device}")
    else:
        logger.info("No GPU is available, using CPU instead")

    logger.info("Loading saved tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained(Path(args["save"], "checkpoints"))
    model = model.to(device)

    model.eval()
    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors="pt")
        new_user_input_ids = new_user_input_ids.to(device)

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        bot_input_ids = bot_input_ids.to(device)

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        chat_history_ids = chat_history_ids.to(device)

        # pretty print last ouput tokens from bot
        print(">> DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True)))


def run(args):

    max_len = int(args["max_len"] / 2)

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(args["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Working on GPU: {device}")
    else:
        logger.info("No GPU is available, using CPU instead")

    ### TOKENIZER
    logger.info(f"Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token

    ### PREPARE DATASET
    data_file = Path(args["save"], "preprocessed", "encoded_dict.pickle")

    if os.path.exists(data_file) and args["prepare"] == False:

        logger.info(f"Loading Saved Dataset ...")

        with open(data_file, "rb") as handle:
            encoded_dict = pickle.load(handle)

        input_ids = encoded_dict["input_ids"]
        attention_masks = encoded_dict["attention_masks"]
        labels = encoded_dict["labels"]

    else:
        data = pd.read_csv(Path("data", "ijcnlp_dailydialog", "train", "dialogues_train.txt"), delimiter="\n", names=["dialogues"])
        data["dialogues"] = data["dialogues"].apply(seputterances)
        logger.info(f"Preparing Dataset ...")
        utterance = []
        history = []

        for i in data.index:
            row = data["dialogues"][i]
            for idx in range(len(row)):
                if idx != 0:
                    utterance.append(row[idx])
                    counter = 1
                    _history = ""
                    for k in range(idx - 1, -1, -1):
                        if counter <= args["context"]:
                            _history = _history + row[k]
                            counter += 1
                        else:
                            break
                        _history = _history + tokenizer.eos_token
                    history.append(_history)
                else:
                    continue

        while True:
            index = random.randint(0, len(history) - 1)
            if len(history[index].split(tokenizer.eos_token)) >= 3:
                logger.info(f"\nExample:\n > {utterance[index]}\n > {history[index]}\n")
                break
            else:
                continue

        ### ENCODING
        logger.info(f"Creating TensorDataset and DataLoader ...")

        input_ids = []
        attention_masks = []
        labels = []
        for i in range(len(utterance)):

            encoded_utterance = tokenizer.encode_plus(
                utterance[i].lower() + tokenizer.eos_token, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
            )

            encoded_history = tokenizer.encode_plus(
                history[i].lower(), max_length=max_len, truncation=True, padding="max_length", return_tensors="pt"
            )

            ids = torch.cat([encoded_utterance["input_ids"][0], encoded_history["input_ids"][0]], dim=0).reshape(1, max_len * 2)
            mask = torch.cat([encoded_utterance["attention_mask"][0], encoded_history["attention_mask"][0]], dim=0).reshape(1, max_len * 2)

            _label = torch.tensor([1 if element != 50256 else -100 for element in encoded_history["input_ids"][0]])
            label = torch.cat([torch.full((max_len,), -100), _label], dim=0).reshape(1, max_len * 2)

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(label)

        encoded_dict = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}

        with open(Path("dialogpt-finetune", "preprocessed", "encoded_dict.pickle"), "wb") as handle:
            pickle.dump(encoded_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args["batch"])

    logger.info(f"Loading the model ...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.to(device)
    logger.info(f"Model size: {sum(t.numel() for t in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    num_training_steps = int(args["epochs"] * len(dataloader))

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    ### TRAINING
    logger.info(f"Training the model ...")
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    tbwriter = SummaryWriter(args["tensorboard"])

    for epoch in range(args["epochs"]):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):

            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            labels = batch[2].to(device)
            inputs = {"input_ids": b_input_ids, "attention_mask": b_attn_mask}
            outputs = model(**inputs, labels=labels)

            optimizer.zero_grad()
            loss = outputs.loss

            running_loss += loss.item()
            if i % 100 == 0:
                logger.info(f"Epoch: {epoch}, Batch: {i}, Loss: {running_loss/100}")

                tbwriter.add_scalar(f"training_loss per epoch: {epoch}, iteration:_loss:", running_loss / 100, i / 100)
                running_loss = 0.0

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        model.save_pretrained(Path(args["save"], "checkpoints"))


if __name__ == "__main__":

    logger.add("logs/{time}.log", format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> <level>{message}</level>", rotation="1 week")

    parser = argparse.ArgumentParser()
    parser.add_argument("-batch", "--batch", help="Batch", required=False, default=4, type=int)
    parser.add_argument("-epochs", "--epochs", help="Training Epochs", required=False, default=3)
    parser.add_argument("-save", "--save", help="Save Checkpoints", required=False, default="dialogpt-finetune")
    parser.add_argument("-lr", "--lr", help="Learning Rate", required=False, default=5e-5)
    parser.add_argument("-clip", "--clip", help="Gradient Clip", required=False, default=2.0)
    parser.add_argument("-seed", "--seed", help="Seed", required=False, default=1234)
    parser.add_argument("-context", "--context", help="Number Context", required=False, default=3)
    parser.add_argument("-max_len", "--max_len", help="Maximum length of tokens", required=False, default=128, type=int)
    parser.add_argument("-prepare", "--prepare", help="Prepare Dataset", required=False, default=False)
    parser.add_argument("-grad_accumulate", "--grad_accumulate", help="Gradient Accumulation", required=False, default=8, type=int)
    parser.add_argument("-tensorboard", "--tensorboard", help="Tensorboard runs", required=False, default="runs/")
    parser.add_argument("-early_stop", "--early_stop", help="Early Stopping", required=False, default=10, type=int)
    parser.add_argument("-eval", "--eval", help="Evaluate", required=False, default=False, type=bool)

    args = vars(parser.parse_args())

    if args["eval"] == True:
        logger.info("Generating Sentences ...")
        evaluate(args)
    else:
        logger.info(f"{args}")
        run(args)
