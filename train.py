import torch
import random

from tqdm.auto import tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler

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


def train(args):

    ### INITIALIZATIONS
    tbwriter = SummaryWriter(args["tensorboard"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Working on device: {device}")

    ### TOKENIZER
    tokenizer, model = load_tokenizer_model()

    ### PREPARE DATA, ENCODING, BATCH
    dataloader = generate_inputs(
        file_path=Path("data", "ijcnlp_dailydialog", "train", "dialogues_train.txt"),
        tokenizer=tokenizer,
        effective_max_len=int(args["max_len"] / 2),
        num_context=args["context"],
        batch_size=args["batch"],
    )
    if args["eval"] == True:
        test_dataloader = generate_inputs(
            file_path=Path("data", "ijcnlp_dailydialog", "test", "dialogues_test.txt"),
            tokenizer=tokenizer,
            effective_max_len=int(args["max_len"] / 2),
            num_context=args["context"],
            batch_size=args["batch"],
        )

    ### OPTIMIZATION
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    num_training_steps = int(args["epochs"] * len(dataloader))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    logger.info(f"Training the model, num_training_steps: {num_training_steps}")
    progress_bar = tqdm(range(num_training_steps))

    model.to(device)
    for epoch in range(args["epochs"]):

        ### TRAINING
        logger.info(f"--------------- Epoch: {epoch} ------------")
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

                tbwriter.add_scalar(f"Training loss: {epoch}:", running_loss / 100, i / 100)
                running_loss = 0.0

            if i % 5000 == 0 and i != 0 and args["eval"] == True:
                model.save_pretrained(Path(args["save"], "checkpoints"))

                ### EVALUATION
                model.eval()
                test_loss = 0

                with torch.no_grad():
                    for i, batch in enumerate(test_dataloader):
                        tb_input_ids = batch[0].to(device)
                        tb_attn_mask = batch[1].to(device)
                        tlabels = batch[2].to(device)

                        inputs = {"input_ids": tb_input_ids, "attention_mask": tb_attn_mask}

                        outputs = model(**inputs, labels=tlabels)

                        loss = outputs.loss

                        test_loss += loss.item()

                    logger.info(f"\nEpoch: {epoch}, Batch: {i}, Testing Loss: {test_loss/ len(test_dataloader)}")
                    tbwriter.add_scalar(f"Testing loss: {epoch}", test_loss / len(test_dataloader), epoch)
                model.train()


if __name__ == "__main__":

    args = argument_parser()
    logger = logging()

    logger.info(f"{args}")

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(args["seed"])

    train(args)
