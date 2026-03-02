"""
Iris Flower Species Classification - Full ML Pipeline
======================================================
Classifies Iris flowers into Setosa, Versicolor, or Virginica
based on sepal and petal measurements.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import joblib

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('IRIS.csv')
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['species'].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("\nStatistical Summary:")
print(df.describe())

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────────
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("\nClasses:", list(le.classes_))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM':                  SVC(probability=True, random_state=42),
    'KNN':                  KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "="*55)
print("MODEL RESULTS")
print("="*55)

for name, model in models.items():
    X_tr = X_train_sc
    X_te = X_test_sc
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    results[name] = {
        'model':    model,
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'test_acc': accuracy_score(y_test, y_pred),
        'y_pred':   y_pred,
    }
    print(f"\n{name}:")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

best_name = max(results, key=lambda k: results[k]['test_acc'])
best = results[best_name]
print(f"\n✅ Best Model: {best_name} — Test Accuracy: {best['test_acc']:.4f}")

# ─────────────────────────────────────────────
# 4. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"CLASSIFICATION REPORT — {best_name}")
print("="*55)
print(classification_report(y_test, best['y_pred'], target_names=le.classes_))

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
sns.set_style('whitegrid')
colors = {'Iris-setosa': '#4CAF50', 'Iris-versicolor': '#2196F3', 'Iris-virginica': '#FF5722'}

fig = plt.figure(figsize=(20, 22))
fig.suptitle('Iris Flower Classification — Analysis & Results',
             fontsize=20, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

# --- Plot 1: Species count ---
ax1 = fig.add_subplot(gs[0, 0])
counts = df['species'].value_counts()
bar_colors = [colors[s] for s in counts.index]
bars = ax1.bar(counts.index, counts.values, color=bar_colors, edgecolor='white', width=0.5)
ax1.set_title('Species Distribution', fontsize=13, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'], rotation=15)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', fontweight='bold')

# --- Plot 2: Sepal scatter ---
ax2 = fig.add_subplot(gs[0, 1])
for species, grp in df.groupby('species'):
    ax2.scatter(grp['sepal_length'], grp['sepal_width'],
                label=species.replace('Iris-', ''), color=colors[species], alpha=0.7, s=50)
ax2.set_title('Sepal: Length vs Width', fontsize=13, fontweight='bold')
ax2.set_xlabel('Sepal Length (cm)')
ax2.set_ylabel('Sepal Width (cm)')
ax2.legend(fontsize=8)

# --- Plot 3: Petal scatter ---
ax3 = fig.add_subplot(gs[0, 2])
for species, grp in df.groupby('species'):
    ax3.scatter(grp['petal_length'], grp['petal_width'],
                label=species.replace('Iris-', ''), color=colors[species], alpha=0.7, s=50)
ax3.set_title('Petal: Length vs Width', fontsize=13, fontweight='bold')
ax3.set_xlabel('Petal Length (cm)')
ax3.set_ylabel('Petal Width (cm)')
ax3.legend(fontsize=8)

# --- Plot 4-7: Feature boxplots ---
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feat_titles = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
positions = [(1,0), (1,1), (1,2), (2,0)]
for feat, title, pos in zip(features, feat_titles, positions):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    data_by_species = [df[df['species']==s][feat].values for s in df['species'].unique()]
    bp = ax.boxplot(data_by_species, patch_artist=True, notch=False)
    for patch, col in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_title(f'{title} by Species', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'], rotation=15, fontsize=8)
    ax.set_ylabel('cm')

# --- Plot 8: Correlation heatmap ---
ax8 = fig.add_subplot(gs[2, 1])
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax8,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax8.set_title('Feature Correlation', fontsize=13, fontweight='bold')
ax8.tick_params(axis='x', rotation=30)

# --- Plot 9: Model comparison ---
ax9 = fig.add_subplot(gs[2, 2])
names = list(results.keys())
accs  = [results[n]['test_acc'] for n in names]
bar_cols = ['#FF7043' if n == best_name else '#42A5F5' for n in names]
bars = ax9.barh(names, accs, color=bar_cols, edgecolor='white')
ax9.set_xlim(0.8, 1.02)
ax9.set_title('Model Test Accuracy', fontsize=13, fontweight='bold')
ax9.set_xlabel('Accuracy')
for bar, val in zip(bars, accs):
    ax9.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)

# --- Plot 10: Confusion matrix (best model) ---
ax10 = fig.add_subplot(gs[3, 0])
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax10,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax10.set_title(f'Confusion Matrix\n({best_name})', fontsize=13, fontweight='bold')
ax10.set_ylabel('Actual')
ax10.set_xlabel('Predicted')
ax10.tick_params(axis='x', rotation=25)

# --- Plot 11: CV scores comparison ---
ax11 = fig.add_subplot(gs[3, 1])
cv_means = [results[n]['cv_mean'] for n in names]
cv_stds  = [results[n]['cv_std']  for n in names]
ax11.barh(names, cv_means, xerr=cv_stds, color=bar_cols, edgecolor='white', capsize=4)
ax11.set_xlim(0.8, 1.05)
ax11.set_title('5-Fold CV Accuracy', fontsize=13, fontweight='bold')
ax11.set_xlabel('Accuracy')

# --- Plot 12: Feature importance (Random Forest) ---
ax12 = fig.add_subplot(gs[3, 2])
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
importances.plot(kind='barh', ax=ax12, color='#42A5F5', edgecolor='white')
ax12.set_title('Feature Importances\n(Random Forest)', fontsize=13, fontweight='bold')
ax12.set_xlabel('Importance')

plt.savefig('iris_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved: iris_analysis.png")

# ─────────────────────────────────────────────
# 6. SAVE MODEL, SCALER & ENCODER
# ─────────────────────────────────────────────
joblib.dump(best['model'], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Saved: best_model.pkl, scaler.pkl, label_encoder.pkl")

# ─────────────────────────────────────────────
# 7. SAVE MODEL COMPARISON CSV
# ─────────────────────────────────────────────
summary = pd.DataFrame([{
    'Model':           n,
    'CV_Accuracy_Mean': f"{results[n]['cv_mean']:.4f}",
    'CV_Accuracy_Std':  f"{results[n]['cv_std']:.4f}",
    'Test_Accuracy':    f"{results[n]['test_acc']:.4f}",
} for n in names])
summary.to_csv('model_comparison.csv', index=False)
print("Saved: model_comparison.csv")

# ─────────────────────────────────────────────
# 8. DEMO — PREDICT NEW FLOWER
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("DEMO — Predict a new flower")
print("="*55)
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                      columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
sample_sc = scaler.transform(sample)
pred = best['model'].predict(sample_sc)
pred_proba = best['model'].predict_proba(sample_sc)[0]
print(f"Input: sepal=5.1x3.5cm, petal=1.4x0.2cm")
print(f"Predicted Species: {le.inverse_transform(pred)[0]}")
print(f"Confidence: {max(pred_proba)*100:.1f}%")
print("\n✅ All done!")
