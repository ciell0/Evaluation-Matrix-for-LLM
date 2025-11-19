# Evaluation-Matrix-for-LLM

# SFT Evaluation Pipeline
Repository ini berisi pipeline evaluasi otomatis untuk model SFT (Supervised Fine-Tuning).  
Pipeline mencakup:

- Inference dataset evaluasi
- Perhitungan banyak metrik (Exact Match, F1, ROUGE-L, BERTScore, Format Validity)
- Logging dan struktur konfigurasi YAML
- Upload otomatis model + hasil evaluasi ke HuggingFace Hub

---

## 📦 Struktur Direktori
your-eval-project/
│
├── configs/
│   ├── eval_config.yaml
│   ├── model_config.yaml
│   └── dataset_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── eval/
│   │   ├── bertscore_eval.py
│   │   ├── exact_match_eval.py
│   │   ├── f1_eval.py
│   │   ├── embedding_similarity_eval.py
│   │   ├── format_validity_eval.py
│   │   └── evaluator.py          # wrapper memanggil semua metrik
│   │
│   ├── inference/
│   │   ├── run_inference.py      # generate output model
│   │   └── tokenizer_loader.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── file_io.py
│   │
│   └── uploader/
│       └── hf_uploader.py
│
├── scripts/
│   ├── run_eval.sh
│   ├── run_inference.sh
│   └── upload_to_hf.sh
│
├── data/
│   ├── test.jsonl
│   └── predictions.jsonl
│
├── results/
│   ├── bertscore.json
│   ├── f1.json
│   ├── exact_match.json
│   ├── embedding_similarity.json
│   ├── format_validity.json
│   └── summary_report.json
│
├── experiment.sh
├── README.md
├── pyproject.toml
└── uv.lock
