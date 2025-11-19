import json
from pathlib import Path

from .bertscore_eval import BertScoreEvaluator
from .exact_match_eval import ExactMatchEvaluator
from .f1_eval import F1ScoreEvaluator
from .embedding_similarity_eval import EmbeddingSimilarityEvaluator
from .format_validity_eval import FormatValidityEvaluator


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.metrics = config["evaluation"]["metrics"]
        self.results_dir = Path(config["paths"]["results_dir"])

        self.eval_map = {
            "bertscore": BertScoreEvaluator(),
            "exact_match": ExactMatchEvaluator(),
            "f1": F1ScoreEvaluator(),
            "embedding_similarity": EmbeddingSimilarityEvaluator(),
            "format_validity": FormatValidityEvaluator(),
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
