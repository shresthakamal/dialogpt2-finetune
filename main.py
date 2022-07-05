import os
import torch
import random
import pickle


from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from options import argument_parser
from utils import logging


def run(args):

    max_len = int(args["max_len"] / 2)

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
        logger.info(f"Preparing Dataset ...")
        data = pd.read_csv(
            Path("data", "ijcnlp_dailydialog", "train", "dialogues_train.txt"), delimiter="\n", names=["dialogues"]
        )
        data["dialogues"] = data["dialogues"].apply(seputterances)

        utterance = []
        history = []

        for i in data.index:
            row = data["dialogues"][i]
            for idx in range(len(row)):
                if idx >= 3:
                    utterance.append(row[idx])
                    counter = 1
                    _history = ""

                    for k in range(idx - args["context"], idx, 1):
                        if counter <= args["context"]:
                            _history = _history + row[k]
                            counter += 1
                        else:
                            break
                        _history = _history + tokenizer.eos_token
                    history.append(_history)

                elif idx != 0 and idx < 3:
                    utterance.append(row[idx])
                    _history = ""
                    for k in range(idx):
                        _history = _history + row[k] + tokenizer.eos_token
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
                utterance[i].lower() + tokenizer.eos_token,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            encoded_history = tokenizer.encode_plus(
                history[i].lower(), max_length=max_len, truncation=True, padding="max_length", return_tensors="pt"
            )

            ids = torch.cat([encoded_utterance["input_ids"][0], encoded_history["input_ids"][0]], dim=0).reshape(
                1, max_len * 2
            )
            mask = torch.cat(
                [encoded_utterance["attention_mask"][0], encoded_history["attention_mask"][0]], dim=0
            ).reshape(1, max_len * 2)

            _label = torch.tensor(
                [element if element != 50256 else -100 for element in encoded_history["input_ids"][0]]
            )
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
    logger.info(f"Length of Dataloader: {len(dataloader)}")

    # Loading the models
    logger.info(f"Loading the model ...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.to(device)
    logger.info(f"Model size: {sum(t.numel() for t in model.parameters())}")

    #   Optimizer and Learning Rates
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
    progress_bar = tqdm(range(num_training_steps))
    tbwriter = SummaryWriter(args["tensorboard"])

    model.train()
    for epoch in range(args["epochs"]):
        running_loss = 0
        for i, batch in enumerate(dataloader):

            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            labels = batch[2].to(device)
            inputs = {"input_ids": b_input_ids, "attention_mask": b_attn_mask}

            optimizer.zero_grad()

            outputs = model(**inputs, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)

            running_loss += loss.item()
            if i % 100 == 0 and i != 0:
                logger.info(f"Epoch: {epoch}, Batch: {i}, Loss: {running_loss/100}")

                tbwriter.add_scalar(f"training_loss per epoch: {epoch}, iteration:_loss:", running_loss / 100, i / 100)
                running_loss = 0.0

        model.save_pretrained(Path(args["save"], "checkpoints"))


if __name__ == "__main__":

    args = argument_parser()
    logger = logging()

    logger.info(f"{args}")

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(args["seed"])

    run(args)
