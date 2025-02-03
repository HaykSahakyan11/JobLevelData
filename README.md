# Transformer-Based Job Title Classification

## 📌 Project Overview

This project aims to classify job titles into predefined categories using multiple machine learning approaches. The dataset consists of job titles along with multiple possible labels. We experimented with three different models:

1. **Transformer-Based Model** (`bert-base-uncased`) - Used for multi-class classification but showed poor results.
2. **Sparse Vector Classifier** - A traditional machine learning model leveraging sparse vector representations of job titles.
3. **FeatureComboMultiLabelModel** - A multi-label classification approach that combines multiple feature sets to improve performance.

While the Transformer-based approach was expected to perform well, **results were poor**. The Sparse Vector Classifier and FeatureComboMultiLabelModel provided alternative perspectives on classification but also faced challenges.

---

## 🚀 Features

✅ **Transformer Model**: Fine-tuned `bert-base-uncased` for job title classification.  
✅ **Sparse Vector Classifier**: Uses traditional sparse vectorization techniques (e.g., TF-IDF) to classify job titles.  
✅ **FeatureComboMultiLabelModel**: Combines different text representations for multi-label classification.  
✅ **Data Preprocessing**: Tokenization using Hugging Face's `AutoTokenizer`, feature extraction using sparse vectors, and label encoding using `LabelEncoder`.  
✅ **Training Pipeline**: Multi-class and multi-label classification approaches using different models.  
✅ **Evaluation Metrics**: Accuracy, F1-score, and recall to assess performance.  
✅ **Inference Pipeline**: Predicts job categories for new job titles.  

---

## 📂 Project Structure

```
project
|-- src
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- data_preprocessing.py
|   |-- model.py
|   |-- train_transformer.py
|   |-- train_sparse_vector_classifier.py
|   |-- train_feature_combo_multilabel.py
|   |-- inference_transformer.py
|   |-- inference_sparse_vector_classifier.py
|   |-- inference_feature_combo_multilabe.py
|-- data
|   |-- JobLevelData.xlsx
|-- models
|   |-- best_model_transformer.pth
|-- requirements.txt
|-- README.md
```

---

## 🛠 Installation & Setup

### 📌 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- tqdm

### 🔧 Setup

```bash
pip install -r requirements.txt
```

---

## 🎯 Training the Model

To train the Transformer model:
```bash
python train_transformer.py
```

To train the Sparse Vector Classifier:
```bash
python train_sparse_vector_classifier.py
```

To train the FeatureComboMultiLabelModel:
```bash
python train_feature_combo_multilabel.py
```

## 🏆 Running Inference

For Transformer-based inference:
```bash
python inference_transformer.py --job_lable "Software Engineer"
```

For Sparse Vector Classifier inference:
```bash
python inference_sparse_vector_classifier.py --job_lable "Software Engineer"
```

For FeatureComboMultiLabelModel inference:
```bash
python inference_feature_combo_multilabel.py --job_lable "Software Engineer"
```

---

## 📊 Results & Challenges

📉 **Transformer model performance was poor**, struggling to correctly classify job titles.  
❗ **Sparse Vector Classifier performed slightly better**, leveraging TF-IDF representations but still showed high confusion.  
💡 **FeatureComboMultiLabelModel provided some improvements**, particularly for multi-label classification, but suffered from overlapping job categories.  

---

## 🔮 Future Work

- 🔄 **Improve multi-label classification strategies** to better capture job title variations.  
- 🏗 **Experiment with alternative Transformer models** (e.g., `roberta-base`, `distilbert`).  
- 🛠 **Enhance feature engineering** in Sparse Vector and FeatureCombo models.  
- 🔍 **Explore ensemble learning** to combine model predictions for better results.  

---

