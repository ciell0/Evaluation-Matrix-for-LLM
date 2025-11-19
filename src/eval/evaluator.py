# src/eval/evaluator.py

import yaml

from .bertscore_eval import BertScoreEvaluator
from .format_validity_eval import FormatValidityEvaluator
from .llm_judge_eval import LLMJudgeEvaluator


class Evaluator:
    def __init__(self, config_path="configs/eval_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.available_metrics = {
            "bertscore": BertScoreEvaluator,
            "format_validity": FormatValidityEvaluator,
            "llm_judge": LLMJudgeEvaluator,        # ← REGISTER METRIK BARU
        }

    def evaluate(self, dataset):
        results = {}

        for metric_name in self.config["metrics"]:
            evaluator_class = self.available_metrics.get(metric_name)
            if evaluator_class is None:
                print(f"[WARN] Unknown metric: {metric_name}")
                continue

            print(f"[Evaluator] Running metric: {metric_name}")
            evaluator = evaluator_class(self.config)
            result = evaluator.compute(dataset)

            results[metric_name] = result

        return results
