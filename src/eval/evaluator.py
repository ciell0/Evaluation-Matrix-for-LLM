import json
from pathlib import Path

from .bertscore_eval import BertScoreEvaluator
from .llm_judge_evaluator import LLMJudgeEvaluator
from .format_validity_eval import FormatValidityEvaluator


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.metrics = config["evaluation"]["metrics"]
        self.results_dir = Path(config["paths"]["results_dir"])
        
        # Mengambil konfigurasi spesifik Judge LLM dari file config
        judge_config = self.config.get("llm_judge", {})

        self.eval_map = {
            "bertscore": BertScoreEvaluator(),
            "format_validity": FormatValidityEvaluator(),
            "llm_judge": LLMJudgeEvaluator(
                judge_model=judge_config.get("model", "gemini-2.5-pro"), 
                num_workers=judge_config.get("workers", 4)
            ),
        }

    def load_jsonl(self, path):
        return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

    def run(self):
        preds = self.load_jsonl(self.config["paths"]["predictions"])
        refs = self.load_jsonl(self.config["paths"]["references"])

        all_results = {}

        for metric in self.metrics:
            print(f"Running metric → {metric}")
            evaluator = self.eval_map[metric]
            result = evaluator.evaluate(preds, refs)
            all_results[metric] = result

        # Save summary
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "summary_report.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print("\nEvaluation completed.")
        print(all_results)

        return all_results
