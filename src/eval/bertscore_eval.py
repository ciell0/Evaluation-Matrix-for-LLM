import json
from bert_score import score as bert_score

class BertScoreEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predictions, references):
        preds = [p["prediction"] for p in predictions]
        refs = [r["output"] for r in references]

        P, R, F1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)

        result = {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
        return result
