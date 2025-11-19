class ExactMatchEvaluator:
    def __init__(self):
        pass

    def normalize(self, s):
        return s.strip().lower()

    def evaluate(self, predictions, references):
        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            if self.normalize(pred["prediction"]) == self.normalize(ref["output"]):
                correct += 1

        return {
            "exact_match": correct / total
        }
