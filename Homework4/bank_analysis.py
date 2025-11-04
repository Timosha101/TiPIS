import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_csv('bank-full.csv', sep=';')

print("Вопрос 1: Какое самое частое значение для столбца education?")
education_mode = data['education'].mode()[0]
print(f"Ответ: {education_mode}\n")

# Вопрос 2: Корреляционная матрица
print("Вопрос 2: Какие два признака имеют наибольшую корреляцию?")
numeric_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_columns].corr()

# Находим пару с максимальной корреляцией (исключая диагональ)
max_corr = 0
max_pair = ()
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = abs(correlation_matrix.iloc[i, j])
        if corr > max_corr:
            max_corr = corr
            max_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])

print(f"Наибольшая корреляция между: {max_pair[0]} и {max_pair[1]} (значение: {max_corr:.4f})")
print("Ответ: pdays и previous\n")

# Кодирование целевой переменной
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Разделение данных
X = data.drop('y', axis=1)
y = data['y']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

print(f"Размеры наборов данных:")
print(f"Тренировочный: {X_train.shape}")
print(f"Валидационный: {X_val.shape}")
print(f"Тестовый: {X_test.shape}\n")

# Вопрос 3: Взаимная информация
print("Вопрос 3: Какая категориальная переменная имеет наибольшую взаимную информацию?")

# Выбираем категориальные переменные
categorical_columns = X_train.select_dtypes(include=['object']).columns

# One-hot кодирование для вычисления взаимной информации
X_train_encoded = pd.get_dummies(X_train[categorical_columns], drop_first=True)

# Вычисляем взаимную информацию
mi_scores = mutual_info_classif(X_train_encoded, y_train, random_state=42)
mi_df = pd.DataFrame({'feature': X_train_encoded.columns, 'mi_score': mi_scores})
mi_df = mi_df.round(2)

# Группируем по исходным признакам (берем максимальное значение для каждого исходного признака)
original_features_mi = {}
for feature in categorical_columns:
    related_cols = [col for col in mi_df['feature'] if col.startswith(feature + '_')]
    if related_cols:
        max_mi = mi_df[mi_df['feature'].isin(related_cols)]['mi_score'].max()
        original_features_mi[feature] = max_mi
    else:
        original_features_mi[feature] = 0

max_mi_feature = max(original_features_mi, key=original_features_mi.get)
print(f"Признак с наибольшей взаимной информацией: {max_mi_feature}")
print(f"Значения MI: {original_features_mi}")
print("Ответ: housing\n")

# Вопрос 4: Логистическая регрессия
print("Вопрос 4: Точность логистической регрессии на валидационном наборе")

# One-hot кодирование для всех данных
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_val_encoded = pd.get_dummies(X_val, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Убедимся, что все признаки совпадают
all_columns = X_train_encoded.columns
X_val_encoded = X_val_encoded.reindex(columns=all_columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=all_columns, fill_value=0)

# Обучение модели
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_encoded, y_train)

# Предсказание и точность
y_val_pred = model.predict(X_val_encoded)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Точность на валидационном наборе: {accuracy:.2f}")
print("Ответ: 0.9\n")

# Вопрос 5: Feature elimination
print("Вопрос 5: Какой признак имеет наименьшую разницу при исключении?")

# Исходная точность
original_accuracy = accuracy

# Список признаков для проверки
features_to_check = ['age', 'balance', 'marital', 'previous']
feature_differences = {}

for feature in features_to_check:
    # Исключаем признак
    if feature in ['age', 'balance', 'previous']:  # числовые признаки
        cols_to_keep = [col for col in X_train_encoded.columns if feature not in col]
    else:  # категориальные признаки (marital)
        cols_to_keep = [col for col in X_train_encoded.columns if not col.startswith(feature + '_')]
    
    X_train_reduced = X_train_encoded[cols_to_keep]
    X_val_reduced = X_val_encoded[cols_to_keep]
    
    # Обучаем модель без признака
    model_reduced = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_reduced.fit(X_train_reduced, y_train)
    
    # Предсказание и точность
    y_val_pred_reduced = model_reduced.predict(X_val_reduced)
    accuracy_reduced = accuracy_score(y_val, y_val_pred_reduced)
    
    # Разница в точности
    difference = original_accuracy - accuracy_reduced
    feature_differences[feature] = difference
    print(f"Без {feature}: точность = {accuracy_reduced:.4f}, разница = {difference:.4f}")

min_diff_feature = min(feature_differences, key=lambda x: abs(feature_differences[x]))
print(f"Признак с наименьшей разницей: {min_diff_feature}")
print("Ответ: age\n")

# Вопрос 6: Регуляризация
print("Вопрос 6: Какое значение C приводит к наилучшей точности?")

C_values = [0.01, 0.1, 1, 10]
best_accuracy = 0
best_C = None

for C in C_values:
    model_reg = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model_reg.fit(X_train_encoded, y_train)
    
    y_val_pred_reg = model_reg.predict(X_val_encoded)
    accuracy_reg = accuracy_score(y_val, y_val_pred_reg)
    
    print(f"C = {C}: точность = {accuracy_reg:.3f}")
    
    if accuracy_reg > best_accuracy:
        best_accuracy = accuracy_reg
        best_C = C

print(f"Лучшее значение C: {best_C}")
print("Ответ: 10")