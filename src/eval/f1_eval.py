import re
from collections import Counter

class F1ScoreEvaluator:
    def __init__(self):
        pass

    def tokenize(self, s):
        return re.findall(r"\w+", s.lower())

    def f1(self, pred, ref):
        pred_tokens = self.tokenize(pred)
        ref_tokens = self.tokenize(ref)

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        common = pred_counter & ref_counter
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, predictions, references):
        f1_scores = []
        for pred, ref in zip(predictions, references):
            f1_scores.append(self.f1(pred["prediction"], ref["output"]))

        return {
            "f1": sum(f1_scores) / len(f1_scores)
        }
