# 🌸 Iris Flower Species Classification

A complete machine learning pipeline to classify Iris flowers into **Setosa**, **Versicolor**, or **Virginica** based on sepal and petal measurements.

## 📁 Project Structure

```
iris-classification/
├── iris_model.py          ← Main ML pipeline script
├── IRIS.csv               ← Dataset (150 samples, 3 species)
├── requirements.txt       ← Python dependencies
├── model_comparison.csv   ← Accuracy table for all models
├── README.md              ← This file
└── .gitignore             ← Git ignore rules
```

## 📊 Dataset
- 150 samples — 50 per species (perfectly balanced)
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: Iris-setosa, Iris-versicolor, Iris-virginica

## 🤖 Model Results

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| **SVM** ✅ | 96.7% ± 1.7% | **96.7%** |
| Logistic Regression | 95.8% ± 2.6% | 93.3% |
| KNN | 95.8% ± 2.6% | 93.3% |
| Random Forest | 95.0% ± 3.1% | 90.0% |
| Gradient Boosting | 95.0% ± 3.1% | 90.0% |

## 🚀 How to Run

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python iris_model.py
```

## 🔮 Predict New Flowers

```python
import joblib, pandas as pd

model   = joblib.load('best_model.pkl')
scaler  = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

prediction = encoder.inverse_transform(model.predict(scaler.transform(new_flower)))
print("Predicted species:", prediction[0])
```
