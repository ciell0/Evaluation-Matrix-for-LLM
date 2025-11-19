import json
from typing import List, Dict
from pathlib import Path
from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger("llm-judge")

class LLMJudgeEvaluator:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.prompt_template = cfg.get("prompt_template", "{context}\n\nJudge the answer: {answer}\nScore (0-1):")
        self.batch_size = cfg.get("batch_size", 8)

    def _make_prompt(self, sample: Dict) -> str:
        context = sample.get("instruction", "") + ("\n" + sample.get("input") if sample.get("input") else "")
        answer = sample.get("prediction")  # predicted by model
        return self.prompt_template.format(context=context, answer=answer)

    def evaluate(self, predictions: List[Dict], references: List[Dict]) -> Dict:
        results = []
        cfg_llm = self.cfg.get("llm", {})
        for i in range(0, len(predictions), self.batch_size):
            batch = predictions[i:i+self.batch_size]
            for pred, ref in zip(batch, references[i:i+self.batch_size]):
                prompt = self._make_prompt({**pred, **ref})
                try:
                    resp = call_llm(prompt, cfg_llm)
                    # parse according to provider response format
                    text = resp.get("text") or resp.get("output") or json.dumps(resp)
                    # naive parse: try extract numeric score
                    score = self._parse_score(text)
                    results.append({"id": pred.get("id"), "score": score, "raw": text})
                except Exception as e:
                    logger.error("LLM judge call failed: %s", e)
                    results.append({"id": pred.get("id"), "score": None, "error": str(e)})

        # aggregate
        scores = [r["score"] for r in results if r.get("score") is not None]
        avg_score = sum(scores)/len(scores) if scores else None
        return {"llm_judge": {"avg_score": avg_score, "per_sample": results}}

    def _parse_score(self, text: str):
        # simple numeric extractor, improve per prompt spec
        import re
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            val = float(m.group(1))
            # normalize if user expects 0-1 or 0-100
            if val > 1:
                return val / 100.0
            return val
        return None
