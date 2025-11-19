from sentence_transformers import SentenceTransformer, util

class EmbeddingSimilarityEvaluator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, predictions, references):
        preds = [p["prediction"] for p in predictions]
        refs = [r["output"] for r in references]

        pred_emb = self.model.encode(preds, convert_to_tensor=True)
        ref_emb = self.model.encode(refs, convert_to_tensor=True)

        cosine_scores = util.cos_sim(pred_emb, ref_emb)
        diagonal_scores = cosine_scores.diag()

        return {
            "embedding_cosine_similarity": float(diagonal_scores.mean())
        }
