from transformers import AutoTokenizer

def load_tokenizer(name: str):
    print(f"[INFO] Loading tokenizer: {name} ...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer
