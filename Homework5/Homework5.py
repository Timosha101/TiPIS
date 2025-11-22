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
df = pd.read_csv('bank/bank-full.csv', sep=';')

# Выбор нужных признаков
selected_features = ['age', 'job', 'marital', 'education', 'balance', 'housing', 
                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
                    'previous', 'poutcome', 'y']
df = df[selected_features]

# Вопрос 1: Самое частое значение для education
education_mode = df['education'].mode()[0]

# Вопрос 2: Корреляционная матрица
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
correlation_matrix = df[numeric_features].corr()

max_corr = 0
max_pair = None
for i in range(len(numeric_features)):
    for j in range(i+1, len(numeric_features)):
        corr_value = abs(correlation_matrix.iloc[i, j])
        if corr_value > max_corr:
            max_corr = corr_value
            max_pair = (numeric_features[i], numeric_features[j])

# Кодирование целевой переменной
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Разделение данных
X = df.drop('y', axis=1)
y = df['y']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Вопрос 3: Взаимная информация
categorical_features = ['job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome']
X_train_categorical = X_train[categorical_features].copy()
for col in categorical_features:
    X_train_categorical[col] = X_train_categorical[col].astype('category').cat.codes

mi_scores = mutual_info_classif(X_train_categorical, y_train, random_state=42)
mi_results = dict(zip(categorical_features, mi_scores))
max_mi_feature = max(mi_results, key=mi_results.get)

# Вопрос 4: Логистическая регрессия
categorical_cols = ['job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome']
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_val_encoded = encoder.transform(X_val[categorical_cols])

X_train_encoded_df = pd.DataFrame(X_train_encoded, 
                                 columns=encoder.get_feature_names_out(categorical_cols),
                                 index=X_train.index)
X_val_encoded_df = pd.DataFrame(X_val_encoded, 
                               columns=encoder.get_feature_names_out(categorical_cols),
                               index=X_val.index)

X_train_final = pd.concat([X_train[numeric_cols], X_train_encoded_df], axis=1)
X_val_final = pd.concat([X_val[numeric_cols], X_val_encoded_df], axis=1)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_final, y_train)

y_val_pred = model.predict(X_val_final)
accuracy = accuracy_score(y_val, y_val_pred)

# Вопрос 5: Feature elimination
base_accuracy = accuracy
features_to_test = ['age', 'balance', 'marital', 'previous']
accuracy_differences = {}

for feature in features_to_test:
    if feature in numeric_cols:
        temp_numeric = [col for col in numeric_cols if col != feature]
        X_train_temp = pd.concat([X_train[temp_numeric], X_train_encoded_df], axis=1)
        X_val_temp = pd.concat([X_val[temp_numeric], X_val_encoded_df], axis=1)
    else:
        cols_to_keep = [col for col in X_train_final.columns if not col.startswith(feature + '_')]
        X_train_temp = X_train_final[cols_to_keep]
        X_val_temp = X_val_final[cols_to_keep]
    
    model_temp = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_temp.fit(X_train_temp, y_train)
    
    y_val_pred_temp = model_temp.predict(X_val_temp)
    temp_accuracy = accuracy_score(y_val, y_val_pred_temp)
    
    difference = base_accuracy - temp_accuracy
    accuracy_differences[feature] = difference

min_diff_feature = min(accuracy_differences, key=lambda x: abs(accuracy_differences[x]))

# Вывод только ответов на вопросы
print("ОТВЕТЫ НА ВОПРОСЫ:")
print(f"Вопрос 1: {education_mode}")
print(f"Вопрос 2: {max_pair[0]} и {max_pair[1]}")
print(f"Вопрос 3: {max_mi_feature}")
print(f"Вопрос 4: {accuracy:.1f}")
print(f"Вопрос 5: {min_diff_feature}")