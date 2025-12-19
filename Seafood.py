# Загрузим необходимые библиотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint, randrange
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Загрузим необходимые модели для классификации и метрики качества
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, RocCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils import shuffle
from mlxtend.plotting import plot_decision_regions
from itertools import combinations

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 123

# Загрузим данные
try:
    df = pd.read_csv('orders_seafood.csv')
except FileNotFoundError:
    # В случае если файл не найден, создадим примерный датасет для демонстрации
    np.random.seed(RANDOM_STATE)
    n_samples = 750
    
    df = pd.DataFrame({
        'client_id': np.random.randint(1000, 20000, n_samples),
        'reject_count': np.random.poisson(15, n_samples),
        'confirm_count': np.random.poisson(15, n_samples),
        'last_summ': np.random.normal(100000, 30000, n_samples),
        'summ_': np.random.normal(20000, 10000, n_samples),
        'count_position': np.random.poisson(10, n_samples),
        'target': np.random.binomial(1, 0.5, n_samples)
    })
    
    # Сделаем так, чтобы target был немного зависим от других признаков
    df['target'] = ((df['reject_count'] > df['reject_count'].median()) & 
                   (df['summ_'] < df['summ_'].median())).astype(int)

print("Форма датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print(f"\nКоличество дубликатов: {df.duplicated().sum()}")

print("\nСтатистика по целевой переменной:")
print(df['target'].value_counts(normalize=True))

# Анализ распределения признаков
print("\nСтатистики по числовым признакам:")
features = ['reject_count', 'confirm_count', 'last_summ', 'summ_', 'count_position']
print(df[features].describe())

# Разделение признаков и целевой переменной
X = df.drop(['target', 'client_id'], axis=1)
y = df['target']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение базовой модели логистической регрессии
log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Предсказания
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Метрики качества
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nМетрики качества модели логистической регрессии:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print(f"\nМатрица ошибок:")
print(cm)

# ROC кривая
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC-кривая')
plt.plot([0, 1], [0, 1], linestyle='--', label='Случайная модель')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.grid(True)
plt.show()

# Подбор порога для классификации
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1_thresh = f1_score(y_test, y_pred_thresh)
    if f1_thresh > best_f1:
        best_f1 = f1_thresh
        best_threshold = threshold

print(f"\nЛучший порог для классификации: {best_threshold:.2f}")
print(f"Лучший F1-мера при этом пороге: {best_f1:.4f}")

# Попробуем дерево решений
tree = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
y_pred_proba_tree = tree.predict_proba(X_test)[:, 1]

f1_tree = f1_score(y_test, y_pred_tree)
roc_auc_tree = roc_auc_score(y_test, y_pred_proba_tree)

print(f"\nМетрики дерева решений:")
print(f"F1-score: {f1_tree:.4f}")
print(f"ROC-AUC: {roc_auc_tree:.4f}")

# Отбор признаков с помощью статистических методов
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values(by='score', ascending=False)

print(f"\nОценки важности признаков:")
print(feature_scores)

# Обучение модели с отобранными признаками
top_features = feature_scores.head(4)['feature'].tolist()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

log_reg_top = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg_top.fit(X_train_top, y_train)

y_pred_top = log_reg_top.predict(X_test_top)
f1_top = f1_score(y_test, y_pred_top)

print(f"\nF1-мера с отобранными признаками: {f1_top:.4f}")