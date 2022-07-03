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

def run(args):

    max_len = int(args["max_len"]/2)
    
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.derterministic = True
    

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])
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

        with open(data_file, 'rb') as handle:
             encoded_dict = pickle.load(handle)

        input_ids = encoded_dict["input_ids"]
        attention_masks=encoded_dict["attention_masks"]
        labels=encoded_dict["labels"]

    else:
        data = pd.read_csv(Path("data", "ijcnlp_dailydialog", "train", "dialogues_train.txt"),  delimiter = "\n", names = ["dialogues"])
        data["dialogues"] = data["dialogues"].apply(seputterances)

        logger.info(f"Preparing Dataset ...")
        utterance = []
        history = []

        for i in data.index:
            row = data["dialogues"][i]

            for idx  in range(len(row)):

                if idx != 0:
                    
                    utterance.append(row[idx])

                    counter = 1
                    _history = ""
                    
                    for k in range(idx-1, -1, -1):
                        if counter <= args["context"]:
                            _history = _history + row[k]
                            counter +=1
                        else:
                            break
                        _history = _history + tokenizer.eos_token
                    history.append(_history)
                else:
                    continue
        
        while True: 
            index = random.randint(0, len(history)-1)
            if len(history[index].split(tokenizer.eos_token))>=3:
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
            
            encoded_utterance = tokenizer.encode_plus(utterance[i].lower() + tokenizer.eos_token, max_length = max_len, padding= "max_length", truncation = True, return_tensors = "pt")
            encoded_history = tokenizer.encode_plus(history[i].lower(), max_length = max_len, truncation = True, padding= "max_length", return_tensors = "pt")

            ids = torch.cat([encoded_utterance["input_ids"][0], encoded_history["input_ids"][0]], dim=0).reshape(1,max_len*2)
            mask = torch.cat([encoded_utterance["attention_mask"][0], encoded_history["attention_mask"][0]], dim=0).reshape(1,max_len*2)
            label = torch.cat([torch.full((max_len,), -100), torch.full((max_len,), 1)], dim = 0).reshape(1, max_len*2)

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(label)

        encoded_dict = {"input_ids": input_ids, "attention_masks": attention_masks, "labels":labels}

        with open(Path("dialogpt-finetune", "preprocessed", 'encoded_dict.pickle'), 'wb') as handle:
            pickle.dump(encoded_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = args["batch"]
        )


    logger.info(f"Loading the model ...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])

    num_training_steps = int(args["epochs"] * len(dataloader))

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )


    ### TRAINING
    logger.info(f"\nTraining the model ...\n")
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    tbwriter = SummaryWriter(args["tensorboard"])

    for epoch in range(args["epochs"]):
        losses = []

        for i, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            labels = batch[2].to(device)
            inputs = {"input_ids": b_input_ids, "attention_mask": b_attn_mask}
            outputs = model(**inputs, labels = labels)
            loss = outputs.loss / args["grad_accumulate"]

            if i%100 == 0:
                losses.append(loss.item())
                logger.info(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")

                tbwriter.add_scalar(f"training_loss per epoch: {epoch}, iteration:_loss:", loss, i/100)

                if len(losses)>=10 and all([True if round(element, 10)==0.0000000000 else False for element in losses[-1:-(10+1):-1] ]):
                    logger.info(f"""{args["early_stop"]} consecutive minimal loss, Epoch Early Skipped.""")
                    break
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),  args["clip"])

            # Gradient Accumulations
            if i%args["grad_accumulate"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)

        model.save_pretrained(Path(args["save"], "checkpoints"))



if __name__ == "__main__":

    logger.add("logs/{time}.log", format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> <level>{message}</level>", rotation="1 week")

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch','--batch', help='Batch', required=False, default=4, type=int)
    parser.add_argument('-epochs','--epochs', help='Training Epochs', required=False, default=3)
    parser.add_argument('-save','--save', help='Save Checkpoints', required=False, default="dialogpt-finetune")
    parser.add_argument('-lr','--lr', help='Learning Rate', required=False, default=5e-5)
    parser.add_argument('-clip','--clip', help='Gradient Clip', required=False, default=2.0)
    parser.add_argument('-seed','--seed', help='Seed', required=False, default=1234)
    parser.add_argument('-context','--context', help='Number Context', required=False, default=3)
    parser.add_argument('-max_len','--max_len', help='Maximum length of tokens', required=False, default=128, type=int)
    parser.add_argument('-prepare','--prepare', help='Prepare Dataset', required=False, default=False)
    parser.add_argument('-grad_accumulate','--grad_accumulate', help='Gradient Accumulation', required=False, default=8, type = int)
    parser.add_argument('-tensorboard','--tensorboard', help='Tensorboard runs', required=False, default="runs/")
    parser.add_argument('-early_stop','--early_stop', help='Early Stopping', required=False, default=10, type = int)

    args = vars(parser.parse_args())

    logger.info(f"{args}")

    run(args)