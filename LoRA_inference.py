import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Paths
BASE_MODEL = r"\home\models\Qwen3-4B-Instruct-2507"
LORA_ADAPTER = r"\home\models\fine-tunes\Qwen3-4B-Instruct_Geocities"

# Generation settings
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1


def load_model_and_tokenizer():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_reply(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


def main():
    model, tokenizer = load_model_and_tokenizer()
    print("\nLoaded Qwen3-4B + LoRA adapter.")
    print("Type an empty line to exit.\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            print("Exiting.")
            break

        print("\nModel>\n")
        reply = generate_reply(model, tokenizer, prompt)
        print(reply)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
