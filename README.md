# Plant Nucleotide Transformer Prediction Tool

A specialized PyTorch implementation for plant genomic sequence classification using the Nucleotide Transformer model, with local prediction capabilities optimized for plant biology research.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Download](#-model-download)
- [Prediction](#-prediction)
- [Input Format](#-input-format)
- [Output Format](#-output-format)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

This project provides specialized tools for plant genomic sequence classification using a fine-tuned Nucleotide Transformer model. The model is optimized for predicting functional elements in plant genomes, trained on diverse plant species data.

**Pre-trained Model Checkpoint:**
- `plant_best_epoch66_auc0.9588.pt` (AUC: 0.9588) - Optimized for plant genomic sequences

## ✨ Features

### 🌱 Plant-Specific Optimizations
- **Multi-species training** on major crop genomes
- **Plant-specific sequence handling** optimized for plant genomic patterns
- **High-accuracy classification** for plant regulatory elements

### 🔬 Technical Features
- 🧬 Optimized for plant genomic sequence classification
- 🌿 Local model loading (no internet required for prediction)
- 📊 Comprehensive prediction outputs with confidence scores
- ⚡ Batch prediction for large plant genomic datasets
- 🎯 High-accuracy classification (AUC > 0.95 on plant datasets)

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ RAM

### Install Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## 📥 Model Download

### Pre-trained Checkpoint
Download the fine-tuned plant model checkpoint:

```bash
# Download plant_best_epoch66_auc0.9588.pt from Google Drive
# Place the file in your project directory
```

### Base Model Setup
The prediction script can automatically download the base Nucleotide Transformer model:

```bash
# Download base model to local directory
python plant_nt_predict.py --download_model
```

This will create a `local_models/` directory containing the model files for offline use.

## 🚀 Quick Start

### 1. Prepare Your Data
Create a CSV file named `input.csv` with your plant sequences:

```csv
sequence
ATCGATCGATCG
GCTAGCTAGCTA
TTTTAAAACCCC
```

### 2. Run Prediction
```bash
python plant_nt_predict.py \
    --checkpoint plant_best_epoch66_auc0.9588.pt \
    --input_csv input.csv \
    --output_csv output.csv \
    --download_model  # Auto-download base model if needed
```

## 🔮 Prediction

### Basic Usage
```bash
python plant_nt_predict.py \
    --checkpoint plant_best_epoch66_auc0.9588.pt \
    --input_csv input.csv \
    --output_csv output.csv
```

### Advanced Options
```bash
python plant_nt_predict.py \
    --checkpoint plant_best_epoch66_auc0.9588.pt \
    --input_csv input.csv \
    --output_csv output.csv \
    --local_model_dir ./local_models/nucleotide-transformer-500m-1000g \
    --batch_size 32 \
    --max_length 512 \
    --device cuda
```

### Command Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to trained model checkpoint (.pt file) |
| `--input_csv` | `input.csv` | Input CSV file with plant sequences |
| `--output_csv` | `output.csv` | Output CSV file for predictions |
| `--local_model_dir` | `./local_models/...` | Local directory containing base model |
| `--batch_size` | 16 | Batch size for prediction |
| `--max_length` | 512 | Maximum sequence length |
| `--device` | cuda | Device for inference (cuda or cpu) |
| `--download_model` | False | Auto-download base model if missing |

## 📊 Input Format

### Required CSV Structure
The input CSV should contain at least one column with plant DNA/RNA sequences:

```csv
sequence
ATCGATCGATCGATCG
GCTAGCTAGCTAGCTA
TTTTAAAACCCCGGGG
```

### Supported Column Names
The script automatically detects sequence columns with these names:
- `sequence`
- `seq` 
- `dna`
- `rna`

Or uses the first column if no matches found.

### Example Input File
```python
import pandas as pd

# Create plant sequence dataset
data = {
    'sequence': [
        'ATCGATCGATCGATCGATCG',  # Plant promoter sequence
        'GCTAGCTAGCTAGCTAGCTA',  # Plant genomic sequence
        'TTTTAAAACCCCGGGGATAT',  # Another plant sequence
    ]
}

df = pd.DataFrame(data)
df.to_csv('input.csv', index=False)
```

## 📈 Output Format

The prediction output `output.csv` includes:

```csv
sequence,prediction,probability_class_0,probability_class_1,confidence,predicted_label
ATCGATCGATCG,1,0.023,0.977,0.977,positive
GCTAGCTAGCTA,0,0.891,0.109,0.891,negative
TTTTAAAACCCC,1,0.156,0.844,0.844,positive
```

### Output Columns Description
| Column | Description |
|--------|-------------|
| `sequence` | Original input sequence |
| `prediction` | Binary prediction (0 or 1) |
| `probability_class_0` | Probability of negative class |
| `probability_class_1` | Probability of positive class |
| `confidence` | Maximum prediction confidence |
| `predicted_label` | Human-readable label (positive/negative) |

### Results Analysis
```python
import pandas as pd

# Load and analyze predictions
results = pd.read_csv('output.csv')

# Summary statistics
print(f"Total plant sequences: {len(results)}")
print(f"Positive predictions: {sum(results['prediction'])}")
print(f"Negative predictions: {len(results) - sum(results['prediction'])}")
print(f"Average confidence: {results['confidence'].mean():.3f}")

# High-confidence predictions
high_conf = results[results['confidence'] > 0.9]
print(f"High-confidence predictions: {len(high_conf)}")
```

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Ensure the checkpoint file exists
ls -la plant_best_epoch66_auc0.9588.pt

# Clear cache and retry
rm -rf ~/.cache/huggingface/transformers/
```

**2. CUDA Out of Memory**
```bash
# Reduce batch size
python plant_nt_predict.py --batch_size 8

# Or use CPU
python plant_nt_predict.py --device cpu
```

**3. Missing Base Model**
```bash
# Auto-download base model
python plant_nt_predict.py --download_model

# Or specify local path
python plant_nt_predict.py --local_model_dir /path/to/local/model
```

### Performance Tips
- Use `--device cuda` for GPU acceleration
- Adjust `--batch_size` based on available GPU memory (8-32 recommended)
- For long plant genomic sequences, increase `--max_length` (up to 1000)
- Use `--download_model` once, then reuse local model for faster startup

## 📊 Performance

The provided plant model checkpoint achieves:
- **AUC: 0.9588**
- **Accuracy: >92%**
- **F1-Score: >0.91**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests related to plant genomics applications.

## 📄 License

This project is for academic and research use. Please check the original Nucleotide Transformer license for commercial use.

## 🙏 Acknowledgments

- InstaDeepAI for the Nucleotide Transformer model
- Hugging Face for the Transformers library  
- The plant genomics community for datasets and tools

---

**Note:** The `plant_best_epoch66_auc0.9588.pt` checkpoint file is available for download via Google Drive. Please contact the maintainers for access.

**For questions and support:**
- 📧 Email: your-email@domain.com
- 💬 Issues: GitHub Issues
- 🐛 Bug Reports: Please include your plant species and sequence length information
