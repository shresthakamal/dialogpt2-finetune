import pandas as pd

from utils import logging

logger = logging()


def seputterances(row):
    try:
        row = row.split("__eou__")
        row = row[:-1]
        return row
    except:
        return row


def prepare_data(file_path, tokenizer, num_context):

    logger.info(f"Preparing Dataset ...")
    data = pd.read_csv(file_path, delimiter="\n", names=["dialogues"])
    data["dialogues"] = data["dialogues"].apply(seputterances)

    utterance = []
    history = []

    for i in data.index:
        row = data["dialogues"][i]
        for idx in range(len(row)):
            if idx >= num_context:
                utterance.append(row[idx])
                counter = 1
                _history = ""

                for k in range(idx - num_context, idx, 1):
                    if counter <= num_context:
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
    return utterance, history
