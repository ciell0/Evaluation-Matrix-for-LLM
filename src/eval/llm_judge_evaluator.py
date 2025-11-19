import os
import json
import numpy as np
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# Pastikan Anda telah menginstal: pip install google-genai numpy

class LLMJudgeEvaluator:
    """
    Kelas Evaluator untuk metrik kualitatif menggunakan LLM-as-a-Judge.
    """
    def __init__(self, judge_model="gemini-2.5-pro", num_workers=4):
        """
        Inisialisasi Judge LLM dan koneksi API.
        """
        # Perlu API Key sebagai environment variable
        api_key = os.environ.get("GEMINI_API_KEY") 
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for LLM Judge.")
        
        self.client = genai.Client(api_key=api_key)
        self.judge_model = judge_model
        self.num_workers = num_workers # Untuk menjalankan evaluasi secara paralel

    def _generate_judge_prompt(self, input_text, reference_answer, model_output):
        """Membangun prompt terstruktur untuk LLM Judge."""
        # Definisi kriteria penilaian spesifik PRD (disarankan untuk disesuaikan!)
        criteria = (
            "Anda adalah penilai ahli. Nilai output model [Output Model] berdasarkan kriteria:\n"
            "1. RELEVANSI (1-5): Relevan dengan [Input]?\n"
            "2. AKURASI (1-5): Seakurat apa dibandingkan [Reference]?\n"
            "3. JUSTIFIKASI (1-5): Seberapa baik model menjelaskan klasifikasinya?\n"
            "Berikan skor dalam format JSON saja: {\"relevancy\": int, \"accuracy\": int, \"justification\": int}"
        )

        prompt = f"""
        {criteria}
        ---
        [Input]: {input_text}
        [Reference]: {reference_answer}
        [Output Model]: {model_output}
        ---
        """
        return prompt.strip()

    def _get_judge_score(self, input_text, reference_answer, model_output):
        """Memanggil API LLM Judge untuk satu sampel dan mengembalikan skor JSON."""
        prompt = self._generate_judge_prompt(input_text, reference_answer, model_output)
        
        try:
            response = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            # Mengasumsikan LLM Judge mengembalikan JSON yang valid
            return json.loads(response.text)
            
        except Exception as e:
            print(f"Error calling Judge API: {e}")
            # Mengembalikan skor 0 jika terjadi error
            return {"relevancy": 0, "accuracy": 0, "justification": 0}

    def evaluate(self, predictions, references):
        """
        Melakukan evaluasi LLM-as-a-Judge secara paralel.
        
        Args:
            predictions (list): List of dicts, e.g., [{"prediction": "...", "input": "..."}]
            references (list): List of dicts, e.g., [{"output": "...", "input": "..."}]

        Returns:
            dict: Rata-rata skor metrik kualitatif.
        """
        
        # Menggabungkan data untuk iterasi
        samples = []
        for p, r in zip(predictions, references):
             samples.append({
                 "input": p.get("input", r.get("input")), # Ambil input dari salah satu sumber
                 "prediction": p["prediction"],
                 "reference": r["output"]
             })

        all_scores = []
        
        # Menggunakan ThreadPoolExecutor untuk menjalankan panggilan API secara paralel
        # Ini penting untuk mempercepat proses evaluasi LLM-as-a-Judge
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for sample in samples:
                future = executor.submit(
                    self._get_judge_score,
                    sample["input"],
                    sample["reference"],
                    sample["prediction"]
                )
                futures.append(future)

            for future in as_completed(futures):
                score = future.result()
                all_scores.append(score)

        # Menghitung rata-rata skor
        avg_relevancy = np.mean([s.get("relevancy", 0) for s in all_scores])
        avg_accuracy = np.mean([s.get("accuracy", 0) for s in all_scores])
        avg_justification = np.mean([s.get("justification", 0) for s in all_scores])
        
        # Format hasil akhir
        result = {
            "judge_relevancy_mean": float(avg_relevancy),
            "judge_accuracy_mean": float(avg_accuracy),
            "judge_justification_mean": float(avg_justification)
        }
        return result
