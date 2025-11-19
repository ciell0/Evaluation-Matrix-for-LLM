# 🚀 SFT Evaluation Framework

**Framework modular untuk melakukan evaluasi model SFT (Supervised Fine-Tuning) secara otomatis**.
Mendukung inference, perhitungan berbagai metrik evaluasi, validasi format output, dan upload hasil ke HuggingFace Hub.

---

# ✨ **Fitur Utama**

### ✅ **1. Inference otomatis**

* Generate prediksi model SFT terhadap dataset test
* Support model local path maupun HuggingFace Hub
* Support dataset lokal maupun HuggingFace Dataset

### ✅ **2. Banyak metrik evaluasi**

Semua metrik dieksekusi melalui modular evaluator:

* **Exact Match (EM)**
* **Token F1 Score**
* **ROUGE-L**
* **BERTScore**
* **Embedding Similarity**
* **Format Validity / Rule Checking**

Metrik dapat di-*enable/disable* melalui file YAML konfigurasi.

### ✅ **3. Script modular**

Termasuk script untuk:

* inference
* evaluasi
* upload ke HuggingFace
* experiment pipeline end-to-end

### ✅ **4. Konfigurasi fleksibel**

Semua komponen (model, dataset, inference config, evaluasi config) ditentukan melalui folder `configs/`.

### ✅ **5. Struktur folder bersih dan scalable**

Repository mengikuti standar proyek ML engineering modern.

---

# 📂 **Struktur Direktori**

```
your-eval-project/
│
├── configs/
│   ├── eval_config.yaml
│   ├── model_config.yaml
│   └── dataset_config.yaml
│
├── src/
│   ├── inference/
│   ├── eval/
│   ├── utils/
│   └── uploader/
│
├── data/
│   ├── test.jsonl
│   └── predictions.jsonl
│
├── scripts/
│   ├── run_inference.sh
│   ├── run_eval.sh
│   └── upload_to_hf.sh
│
├── experiment.sh
├── pyproject.toml
└── README.md
```

---

# 🔧 **Persiapan Lingkungan**

Repo ini menggunakan **Python 3.10+** dan dependency dikelola menggunakan **uv** atau pip.

## 📥 Instalasi menggunakan uv (direkomendasikan)

```bash
uv sync
```

## 📥 Instalasi menggunakan pip

```bash
pip install -r requirements.txt
```

---

# 🔌 **Konfigurasi**

Semua konfigurasi ada dalam folder:

```
configs/
```

### 🧩 `model_config.yaml`

Mengatur model yang dipakai untuk inference:

```yaml
model_name: my-sft-model
tokenizer_name: my-sft-model
device: cuda
```

### 🧩 `dataset_config.yaml`

Mengatur sumber dataset:

```yaml
dataset_path: data/test.jsonl
```

Bisa juga memakai HuggingFace:

```yaml
dataset_path: hf://username/my-dataset
```

### 🧩 `eval_config.yaml`

Menentukan metrik apa saja yang dipakai:

```yaml
metrics:
  exact_match: true
  f1: true
  rouge: true
  bertscore: false
  embed_sim: true
  format_validity: true

generation:
  max_new_tokens: 128
  temperature: 0.7
```

---

# ▶️ **Menjalankan Proyek**

Ada 3 cara menjalankan pipeline: manual, per step, atau otomatis.

---

## **1. Menjalankan inference**

```bash
bash scripts/run_inference.sh
```

Atau:

```bash
python -m src.inference.run_inference \
  --model_path outputs/my-sft-model \
  --data_path data/test.jsonl \
  --output_path results/predictions.jsonl
```

---

## **2. Menjalankan evaluasi**

```bash
bash scripts/run_eval.sh
```

Atau:

```bash
python -m src.eval.evaluator \
  --pred_path results/predictions.jsonl \
  --gold_path data/test.jsonl \
  --config configs/eval_config.yaml \
  --output_path results/metrics.json
```

---

## **3. Upload ke HuggingFace Hub**

```bash
bash scripts/upload_to_hf.sh
```

---

# 🚀 **Menjalankan Pipeline End-to-End**

Jalankan satu perintah:

```bash
bash experiment.sh
```

Pipeline akan menjalankan:

1. inference
2. evaluasi
3. upload hasil

---

# 📊 **Output Evaluasi**

Hasil evaluasi disimpan di:

```
results/metrics.json
```

Contoh output:

```json
{
  "exact_match": 0.32,
  "f1_score": 0.71,
  "rouge_l": 0.56,
  "embedding_similarity": 0.83,
  "format_validity": 0.92
}
```

---

# 🔧 **Menambah Metrik Baru**

Tambah file baru di:

```
src/eval/
```

Contoh:

```
my_custom_metric.py
```

Daftarkan ke evaluator pada:

```
src/eval/evaluator.py
```

Framework ini memang dibuat untuk memudahkan extend metrik baru.

---
