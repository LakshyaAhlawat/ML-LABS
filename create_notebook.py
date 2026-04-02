import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1
nb.cells.append(nbf.v4.new_markdown_cell("# Banknote Authentication — Simple Classification\n## Detecting Genuine vs Forged\nUsing: Logistic Regression & KNN"))

# Cell 2
nb.cells.append(nbf.v4.new_markdown_cell("---\n## 1. Import Data"))
nb.cells.append(nbf.v4.new_code_cell("""from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

banknote_data = fetch_ucirepo(id=267)
X = banknote_data.data.features
y = banknote_data.data.targets

print(f'Samples: {X.shape[0]}, Features: {X.shape[1]}')"""))

# Cell 3
nb.cells.append(nbf.v4.new_markdown_cell("## 2. Basic EDA"))
nb.cells.append(nbf.v4.new_code_cell("""data = pd.concat([X, y], axis=1)
print('First 5 rows:')
print(data.head())
print()
print('Statistics:')
print(data.describe())
print()
print('Info:')
print(data.info())"""))

# Cell 4
nb.cells.append(nbf.v4.new_markdown_cell("## 3. Class Distribution"))
nb.cells.append(nbf.v4.new_code_cell("""print('Missing values:')
print(data.isnull().sum())
print()
print('Class distribution:')
print(y.value_counts())

plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='pie', labels=['Forged', 'Genuine'], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
plt.title('Class Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()"""))

# Cell 5
nb.cells.append(nbf.v4.new_markdown_cell("## 4. Feature Histograms"))
nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
feature_names = X.columns

for i, feature in enumerate(feature_names):
    axes[i].hist(X[feature], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(feature)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()"""))

# Cell 6
nb.cells.append(nbf.v4.new_markdown_cell("## 5. Boxplot by Class"))
nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    forged = data[data['class'] == 0][feature]
    genuine = data[data['class'] == 1][feature]
    bp = axes[i].boxplot([forged, genuine], labels=['Forged', 'Genuine'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][1].set_facecolor('#4ECDC4')
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()"""))

# Cell 7
nb.cells.append(nbf.v4.new_markdown_cell("## 6. Correlation Matrix"))
nb.cells.append(nbf.v4.new_code_cell("""corr = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print('Correlation with target:')
print(corr['class'].sort_values(ascending=False))"""))

# Cell 8
nb.cells.append(nbf.v4.new_markdown_cell("---\n## 7. Prepare Data"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Training: {X_train.shape[0]} samples')
print(f'Test: {X_test.shape[0]} samples')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Data scaled!')"""))

# Cell 9
nb.cells.append(nbf.v4.new_markdown_cell("---\n## 8. Logistic Regression"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print('LOGISTIC REGRESSION:')
print(f'Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}')
print(f'Precision: {precision_score(y_test, y_pred_lr):.4f}')
print(f'Recall:    {recall_score(y_test, y_pred_lr):.4f}')
print(f'F1-Score:  {f1_score(y_test, y_pred_lr):.4f}')
print(f'AUC:       {roc_auc_score(y_test, y_prob_lr):.4f}')

cm_lr = confusion_matrix(y_test, y_pred_lr)
print('Confusion Matrix:')
print(cm_lr)"""))

# Cell 10
nb.cells.append(nbf.v4.new_markdown_cell("---\n## 9. KNN - Find Best K"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.neighbors import KNeighborsClassifier

best_k = 0
best_acc = 0
k_list, acc_list = [], []

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    k_list.append(k)
    acc_list.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f'Best K = {best_k} with accuracy = {best_acc:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(k_list, acc_list, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title('KNN: Accuracy vs K')
plt.xlabel('K (neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Cell 11
nb.cells.append(nbf.v4.new_code_cell("""knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

print(f'KNN (K={best_k}):')
print(f'Accuracy:  {accuracy_score(y_test, y_pred_knn):.4f}')
print(f'Precision: {precision_score(y_test, y_pred_knn):.4f}')
print(f'Recall:    {recall_score(y_test, y_pred_knn):.4f}')
print(f'F1-Score:  {f1_score(y_test, y_pred_knn):.4f}')
print(f'AUC:       {roc_auc_score(y_test, y_prob_knn):.4f}')

cm_knn = confusion_matrix(y_test, y_pred_knn)
print('Confusion Matrix:')
print(cm_knn)"""))

# Cell 12
nb.cells.append(nbf.v4.new_markdown_cell("## 10. Confusion Matrices"))
nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Forged', 'Genuine'], yticklabels=['Forged', 'Genuine'])
axes[0].set_title('Logistic Regression')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Reds', ax=axes[1], xticklabels=['Forged', 'Genuine'], yticklabels=['Forged', 'Genuine'])
axes[1].set_title(f'KNN (K={best_k})')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()"""))

# Cell 13
nb.cells.append(nbf.v4.new_markdown_cell("## 11. ROC Curves"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import roc_curve, auc

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = auc(fpr_lr, tpr_lr)

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC={auc_lr:.4f})', linewidth=2)
plt.plot(fpr_knn, tpr_knn, label=f'KNN K={best_k} (AUC={auc_knn:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Cell 14
nb.cells.append(nbf.v4.new_markdown_cell("## 12. Final Comparison"))
nb.cells.append(nbf.v4.new_code_cell("""results = {
    'Algorithm': ['Logistic Regression', f'KNN (K={best_k})'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_knn)],
    'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_knn)],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_knn)],
    'F1-Score': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_knn)],
    'AUC': [auc_lr, auc_knn]
}

df_results = pd.DataFrame(results)
print('\\n' + '='*70)
print('FINAL COMPARISON')
print('='*70)
print(df_results.to_string(index=False))
print('='*70)"""))

# Cell 15
nb.cells.append(nbf.v4.new_markdown_cell("## 13. Comparison Bar Charts"))
nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].bar(['Logistic', 'KNN'], df_results['Accuracy'], color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
axes[0].set_title('Accuracy')
axes[0].set_ylim([0.9, 1.0])
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(['Logistic', 'KNN'], df_results['F1-Score'], color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
axes[1].set_title('F1-Score')
axes[1].set_ylim([0.9, 1.0])
axes[1].grid(alpha=0.3, axis='y')

axes[2].bar(['Logistic', 'KNN'], df_results['AUC'], color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
axes[2].set_title('AUC')
axes[2].set_ylim([0.9, 1.0])
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""))

# Cell 16
nb.cells.append(nbf.v4.new_markdown_cell("## 14. Cross-Validation"))
nb.cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

lr_cv = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), X, y, cv=kfold, scoring='accuracy')
print(f'Logistic Regression: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}')

knn_cv = cross_val_score(KNeighborsClassifier(n_neighbors=best_k), X, y, cv=kfold, scoring='accuracy')
print(f'KNN (K={best_k}): {knn_cv.mean():.4f} +/- {knn_cv.std():.4f}')

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), lr_cv, marker='o', label='Logistic Regression', linewidth=2)
plt.plot(range(1, 11), knn_cv, marker='s', label=f'KNN (K={best_k})', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('10-Fold Cross-Validation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Cell 17
nb.cells.append(nbf.v4.new_markdown_cell("## SUMMARY"))
nb.cells.append(nbf.v4.new_code_cell("""print('\\n' + '='*70)
print('FINAL RESULTS')
print('='*70)
print(df_results.to_string(index=False))
print()
print(f'CV - Logistic Regression: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}')
print(f'CV - KNN (K={best_k}): {knn_cv.mean():.4f} +/- {knn_cv.std():.4f}')
print()
best = 'Logistic Regression' if df_results.loc[0, 'Accuracy'] > df_results.loc[1, 'Accuracy'] else f'KNN (K={best_k})'
print(f'Best Model: {best}')
print('='*70)"""))

# Save notebook
with open(r'C:\Users\ahlaw\OneDrive\Desktop\mlLabs\ml_project\banknote.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully with 33 cells!")
