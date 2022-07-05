import torch
import random

from tqdm.auto import tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from options import argument_parser
from make_dataset import prepare_data
from generate_inputs import generate_inputs
from utils import logging


def load_tokenizer_model():
    logger.info(f"Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    logger.info(f"Pre-Trained Model loaded, {sum(t.numel() for t in model.parameters())} parameters")
    return tokenizer, model


def run(args):

    ### INITIALIZATIONS
    tbwriter = SummaryWriter(args["tensorboard"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Working on GPU: {device}")
    else:
        logger.info("No GPU is available, using CPU instead")

    ### TOKENIZER
    tokenizer, model = load_tokenizer_model()

    ### PREPARE DATA
    utterance, history = prepare_data(
        file_path=Path("data", "ijcnlp_dailydialog", "train", "dialogues_train.txt"),
        tokenizer=tokenizer,
        num_context=args["context"],
    )

    ### ENCODING
    input_ids, attention_masks, labels = generate_inputs(
        utterance, history, tokenizer, effective_max_len=int(args["max_len"] / 2)
    )
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args["batch"])

    #   Optimizer and Learning Rates
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    num_training_steps = int(args["epochs"] * len(dataloader))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    logger.info(f"Training the model ..., num_training_steps: {num_training_steps}")
    progress_bar = tqdm(range(num_training_steps))

    ###

    model.to(device)
    for epoch in range(args["epochs"]):

        ### TRAINING
        model.train()
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

        ### EVALUATION


if __name__ == "__main__":

    args = argument_parser()
    logger = logging()

    logger.info(f"{args}")

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(args["seed"])

    run(args)
