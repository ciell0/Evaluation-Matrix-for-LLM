import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from .tokenizer_loader import load_tokenizer


def run_inference(model_name, dataset_path, output_path):

    # Load dataset
    dataset = [json.loads(line) for line in open(dataset_path, "r")]

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    results = []

    for item in dataset:
        q = item["question"]

        prompt = f"Jawablah pertanyaan berikut secara akurat dan sesuai konteks:\n\n{q}\n\nJawaban:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=300)

        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": item["question"],
            "reference": item["answer"],
            "output": model_answer
        })

    # save
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results
