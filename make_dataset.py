import os
import pickle
import pandas as pd
from pathlib import Path

from utils import logging

logger = logging()


def seputterances(row):
    try:
        row = row.split("__eou__")
        row = row[:-1]
        return row
    except:
        return row


def prepare_data():
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
