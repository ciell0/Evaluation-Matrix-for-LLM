import yaml
import json
from tokenizer_loader import load_tokenizer
from transformers import AutoModelForCausalLM

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_config):
    model_name = model_config["model_name"]
    device = model_config.get("device", "cpu")

    print(f"[INFO] Loading model: {model_name} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model

def generate_output(model, tokenizer, prompt, gen_config):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen_config.get("max_new_tokens", 128),
        temperature=gen_config.get("temperature", 0.7),
        top_p=gen_config.get("top_p", 0.9),
        do_sample=gen_config.get("do_sample", True)
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load configs
    eval_cfg = load_yaml("../configs/eval_config.yaml")
    model_cfg = load_yaml("../configs/model_config.yaml")
    dataset_cfg = load_yaml("../configs/dataset_config.yaml")

    # Load model + tokenizer
    tokenizer = load_tokenizer(model_cfg["tokenizer_name"])
    model = load_model(model_cfg)

    # Load dataset
    dataset_path = dataset_cfg["dataset_path"]
    with open(dataset_path, "r") as f:
        dataset = [json.loads(l) for l in f]

    print(f"[INFO] Loaded {len(dataset)} items.")

    results = []
    for sample in dataset:
        prompt = sample["input"]
        gen_out = generate_output(model, tokenizer, prompt, eval_cfg["generation"])
        results.append({"input": prompt, "output": gen_out})

    # Save result
    output_path = eval_cfg["output_path"]
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[INFO] Saved inference output to {output_path}")
