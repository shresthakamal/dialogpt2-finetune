from logging import handlers
from config import *

import pandas as pd
import pickle

def seputterances(row):
    try:
        row = row.split("__eou__")
        row = row[:-1]
        return row
    except:
        return row

def make_dataset(filename):

    dialogues = pd.read_csv(Path(DATA_DIR, "train","dialogues_train.txt"),  delimiter = "\n", names = ["dialogues"])

    dialogues["dialogues"] = dialogues["dialogues"].apply(seputterances)

    dataset = {"S1":[], "S2": []}

    for i in range(len(dialogues)):
        for j in range(len(dialogues["dialogues"][i])):
            dataset["S1"].append(dialogues["dialogues"][i][j])
            try:
                dataset["S2"].append(dialogues["dialogues"][i][j+1])
            except:
                dataset["S2"].append("EOS")

    with open(Path("data", filename), 'wb') as handler:
        pickle.dump(dataset, handler, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    make_dataset("inturn_conversations.pkl")