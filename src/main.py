import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler



def seputterances(row):
    try:
        row = row.split("__eou__")
        row = row[:-1]
        return row
    except:
        return row

def run():

    SEED  = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.derterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
        print(f"[INFO]: Working on GPU: {device}")
    else:
        print("[INFO]: No GPU is available, using CPU instead")


    data = pd.read_csv(Path("data", "ijcnlp_dailydialog", "train", "dialogues_train.txt"),  delimiter = "\n", names = ["dialogues"])

    data["dialogues"] = data["dialogues"].apply(seputterances)

    
    print(f"[INFO]: Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    num_context = 3

    print(f"[INFO]: Preparing Dataset ...")
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
                    if counter <= num_context:
                        _history = _history + row[k]
                        counter +=1
                    else:
                        break
                    _history = _history + tokenizer.eos_token
                history.append(_history)
            else:
                continue
        
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    max_len = 32

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
    
    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"[INFO]: Creating TensorDataset and DataLoader ...")
    dataset = TensorDataset(input_ids, attention_masks, labels)

    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = 4
        )


    print(f"[INFO]: Loading the model ...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3

    num_training_steps = int(num_epochs * len(dataloader))

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # progress_bar = tqdm(range(num_training_steps))

    model.train()
    print(f"Training the model ...")
    
    for i, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        labels = batch[2].to(device)

        inputs = {"input_ids": b_input_ids, "attention_mask": b_attn_mask}

        outputs = model(**inputs, labels = labels)

        print(outputs)
        break

    # for epoch in range(num_epochs):
    #     for i in range(len(tokenized_dataset["input_ids"])):
    #         inputs= {}

    #         ids = tokenized_dataset["input_ids"][:num_batch]
    #         mask = tokenized_dataset["attention_mask"][:num_batch]

    #         inputs["input_ids"] = ids.to(device)
    #         inputs["attention_mask"] = mask.to(device)

    #         outputs = model(**inputs, labels = inputs["input_ids"])

    #         loss = outputs.loss/n

    #         loss.backward()

    #         if (i+1) % 8 == 0:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
            
    #         progress_bar.update(1)
    #     model.save_pretrained("dialogpt-finetne")

if __name__ == "__main__":
    run()