import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm
import os.path, pickle
import random


SEED  = 1234

random.seed(SEED)
torch.mannual_seed(SEED)
torch.backends.cudnn.derterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    print("[INFO]: No GPU is available, using CPU instead")


DATA_DIR = Path("data", "ijcnlp_dailydialog")

def seputterances(row):
    try:
        row = row.split("__eou__")
        row = row[:-1]
        return row
    except:
        return row

def run():

    if os.path.isfile(Path("data", "inturn_conversations.pkl")):
        print(f"[INFO]: Loading saved tokens...")

        with open(Path("data", "inturn_conversations.pkl"), 'rb') as handle:
            tokenized_dataset = pickle.load(handle)
    else:
        print(f"[INFO]: Saved Tokens not found, creating tokens ...")

        data = pd.read_csv(Path(DATA_DIR, "dialogues_text.txt"),  delimiter = "\n", names = ["dialogues"])

        data["dialogues"] = data["dialogues"].apply(seputterances)

        dataset = {"S1":[], "S2": []}

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

        for i in range(len(data)):
            for j in range(len(data["dialogues"][i])):
                dataset["S1"].append(data["dialogues"][i][j])
                try:
                    dataset["S2"].append(data["dialogues"][i][j+1])
                except:
                    dataset["S2"].append(tokenizer.eos_token)
        
        tokenizer.pad_token = tokenizer.eos_token

        tokenized_dataset = tokenizer(
            dataset["S1"],
            dataset["S2"],
            padding=True,
            truncation=True,
            max_length = 128,
            return_tensors = "pt"
        )

        with open(Path("data", "inturn_conversations.pkl"), 'wb') as handle:
            pickle.dump(tokenized_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO]: Loading the model ...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3

    num_batch = 2

    num_training_steps = int(num_epochs * len(tokenized_dataset["input_ids"])/4)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(num_epochs):
        for i in range(len(tokenized_dataset["input_ids"])):
            inputs= {}

            ids = tokenized_dataset["input_ids"][:num_batch]
            mask = tokenized_dataset["attention_mask"][:num_batch]

            inputs["input_ids"] = ids.to(device)
            inputs["attention_mask"] = mask.to(device)

            outputs = model(**inputs, labels = inputs["input_ids"])

            loss = outputs.loss/n

            loss.backward()

            if (i+1) % 8 == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
        model.save_pretrained("dialogpt-finetne")

if __name__ == "__main__":
    run()