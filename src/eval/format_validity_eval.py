import json

class FormatValidityEvaluator:
    def __init__(self):
        pass

    def is_valid_json(self, text):
        try:
            json.loads(text)
            return True
        except:
            return False

    def evaluate(self, predictions, references=None):
        total = len(predictions)
        valid = 0

        for p in predictions:
            if self.is_valid_json(p["prediction"]):
                valid += 1

        return {
            "valid_json_rate": valid / total
        }
