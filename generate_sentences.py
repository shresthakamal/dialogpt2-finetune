from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_sentences(model, step=5, max_len=1000):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained(model)

    for step in range(step):
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors="pt")

        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        )

        chat_history_ids = model.generate(bot_input_ids, max_length=max_len, pad_token_id=tokenizer.eos_token_id)

        print(
            "DialoGPT: {}".format(
                tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True)
            )
        )


if __name__ == "__main__":
    generate_sentences(model="dialogpt-finetune/checkpoints")
