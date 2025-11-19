# src/eval/llm_judge_eval.py

import os
import google.generativeai as genai


class LLMJudgeEvaluator:
    def __init__(self, config):
        self.model_name = config["llm_judge"]["model"]
        self.prompt_template = config["llm_judge"]["scoring_prompt"]
        
        api_key_env = config["llm_judge"].get("api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)

        if api_key is None:
            raise ValueError(f"[LLMJudgeEvaluator] Missing API key in environment variable: {api_key_env}")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def score_single(self, instruction, reference_answer, model_output):
        """Meminta Gemini menilai output model"""

        prompt = f"""
{self.prompt_template}

Instruction:
{instruction}

Reference (Ground Truth):
{reference_answer}

Model Output:
{model_output}

Score (1-5):
"""

        response = self.model.generate_content(prompt)

        try:
            score = float(response.text.strip())
        except Exception:
            score = 1.0   # fallback

        return score

    def compute(self, dataset):
        """
        dataset: list of dict {instruction, reference, output}
        """
        scores = []
        for item in dataset:
            score = self.score_single(
                instruction=item["instruction"],
                reference_answer=item["reference"],
                model_output=item["output"]
            )
            scores.append(score)

        return {
            "llm_judge_avg": sum(scores) / len(scores),
            "llm_judge_all_scores": scores
        }
