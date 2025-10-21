import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import joblib
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('diamonds_train.csv')

# Анализ данных
print("Информация о данных:")
df.info()

print("\nСтатистика числовых признаков:")
print(df.describe())

# Предварительная обработка данных
print("\n=== ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ ===")

# Проверяем пропущенные значения
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Пропущенные значения:")
    display(missing[missing > 0])
else:
    print("✓ Пропущенных значений нет")

# Проверяем категориальные переменные
categorical_cols = ['cut', 'color', 'clarity']
print("\nКатегориальные переменные:")
for col in categorical_cols:
    if col in df.columns:
        unique_vals = sorted(df[col].unique())
        print(f"{col}: {unique_vals} (количество: {len(unique_vals)})")

# Кодируем категориальные переменные
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"✓ Закодирована переменная: {col}")

# Подготовка признаков
print("\n=== ПОДГОТОВКА ПРИЗНАКОВ ===")

# Определяем признаки для обучения (исключаем исходные категориальные колонки)
feature_columns = [col for col in df.columns 
                  if col != 'price' and col not in categorical_cols]

print(f"Признаки для обучения ({len(feature_columns)}): {feature_columns}")

# Подготавливаем X и y
X = df[feature_columns]
y = df['price']

print(f"Размеры данных: X {X.shape}, y {y.shape}")
print(f"Статистика цены - среднее: ${y.mean():,.2f}, медиана: ${y.median():,.2f}, std: ${y.std():,.2f}")

# Разделение на train/test
print("\n=== РАЗДЕЛЕНИЕ ДАННЫХ ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

print(f"Обучающая выборка: {X_train.shape[0]:,} образцов ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Тестовая выборка: {X_test.shape[0]:,} образцов ({X_test.shape[0]/len(X)*100:.1f}%)")

# Нормализация и стандартизация данных
print("\n=== ПРЕОБРАЗОВАНИЕ ДАННЫХ ===")

# Создаем копии для безопасного преобразования
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Этап 1: Нормализация (приводим к диапазону [0,1])
minmax_scaler = MinMaxScaler()
X_train_norm = minmax_scaler.fit_transform(X_train_processed)
X_test_norm = minmax_scaler.transform(X_test_processed)
print("✓ Нормализация MinMaxScaler завершена")

# Этап 2: Стандартизация нормализованных данных (μ=0, σ=1)
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train_norm)
X_test_scaled = standard_scaler.transform(X_test_norm)
print("✓ Стандартизация StandardScaler завершена")

# Обучение модели
print("\n=== ОБУЧЕНИЕ МОДЕЛИ ===")

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✓ Модель LinearRegression обучена")

# Предсказания
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Оценка качества модели
print("\n=== ОЦЕНКА КАЧЕСТВА МОДЕЛИ ===")

# Основные метрики
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("МЕТРИКИ КАЧЕСТВА:")
print(f"R² Score - Train: {train_r2:.4f}, Test: {test_r2:.4f}")
print(f"RMSE - Train: ${train_rmse:,.2f}, Test: ${test_rmse:,.2f}")

# Кросс-валидация
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Кросс-валидация R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

overfitting = train_r2 - test_r2
print(f"Переобучение (разница R²): {overfitting:.4f}")

# Анализ коэффициентов
print("\n=== АНАЛИЗ МОДЕЛИ ===")

print(f"Перехват (intercept): {model.intercept_:.2f}")

# Коэффициенты по важности (абсолютное значение)
coef_importance = [(abs(coef), feature, coef) for feature, coef in zip(feature_columns, model.coef_)]
coef_importance.sort(reverse=True)

print("Коэффициенты по важности (абсолютные значения):")
for i, (abs_coef, feature, coef) in enumerate(coef_importance, 1):
    sign = "+" if coef >= 0 else "-"
    print(f"{i:2d}. {feature:15}: {sign} {abs(coef):10.4f}")

# Полное уравнение
equation_parts = [f"{model.intercept_:.2f}"]
for feature, coef in zip(feature_columns, model.coef_):
    sign = "+" if coef >= 0 else "-"
    equation_parts.append(f"{sign} {abs(coef):.4f}*{feature}")

equation = "price = " + " ".join(equation_parts)
print(f"\nУРАВНЕНИЕ РЕГРЕССИИ:\n{equation}")

# Финальная модель для соревнований
print("\n=== ФИНАЛЬНАЯ МОДЕЛЬ ===")

# Подготовка всех данных
X_final = df[feature_columns]
y_final = df['price']

# Преобразование данных
minmax_scaler_final = MinMaxScaler()
X_final_norm = minmax_scaler_final.fit_transform(X_final)

standard_scaler_final = StandardScaler()
X_final_scaled = standard_scaler_final.fit_transform(X_final_norm)

# Обучение финальной модели
final_model = LinearRegression()
final_model.fit(X_final_scaled, y_final)

# Оценка финальной модели
final_predictions = final_model.predict(X_final_scaled)
final_r2 = r2_score(y_final, final_predictions)
final_rmse = np.sqrt(mean_squared_error(y_final, final_predictions))

print(f"Финальная модель - R²: {final_r2:.4f}, RMSE: ${final_rmse:,.2f}")

# Сохранение всех компонентов
joblib.dump(final_model, 'final_model.pkl')
joblib.dump(minmax_scaler_final, 'minmax_scaler.pkl')
joblib.dump(standard_scaler_final, 'standard_scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')
print("✓ Все компоненты модели сохранены")

# Тестирование и создание файла для соревнований
print("\n=== ПОДГОТОВКА РЕЗУЛЬТАТОВ ===")

# Загрузка тестовых данных
df_test = pd.read_csv('diamonds_test.csv')
print(f"Тестовые данные: {df_test.shape[0]:,} образцов")

# Сохраняем id для submission
test_ids = df_test['id'].copy()

# Кодируем категориальные переменные в тестовых данных
for col in ['cut', 'color', 'clarity']:
    # Проверяем наличие неизвестных категорий
    unknown_categories = set(df_test[col].unique()) - set(label_encoders[col].classes_)
    if unknown_categories:
        print(f"⚠ Предупреждение: неизвестные категории в {col}: {unknown_categories}")
        # Для неизвестных категорий используем наиболее частую категорию
        most_frequent = label_encoders[col].classes_[0]
        df_test[col] = df_test[col].apply(lambda x: x if x in label_encoders[col].classes_ else most_frequent)
    
    df_test[col + '_encoded'] = label_encoders[col].transform(df_test[col])

# Подготовка признаков тестовых данных
X_test_final = df_test[feature_columns]

# Преобразование тестовых данных (используем обученные scaler'ы)
X_test_norm = minmax_scaler_final.transform(X_test_final)
X_test_scaled = standard_scaler_final.transform(X_test_norm)

# Предсказание
predictions = final_model.predict(X_test_scaled)

# Проверка предсказаний
print(f"Статистика предсказаний: min=${predictions.min():.2f}, max=${predictions.max():.2f}, mean=${predictions.mean():.2f}")

# Сохранение результатов
df_result = pd.DataFrame({
    'id': df_test['id'], 
    'price': predictions
})

# Проверяем, что нет отрицательных цен
if (df_result['price'] < 0).any():
    print("⚠ Предупреждение: обнаружены отрицательные цены, заменяем на минимальную положительную")
    min_positive_price = df_result[df_result['price'] > 0]['price'].min()
    df_result['price'] = df_result['price'].clip(lower=min_positive_price)

df_result.to_csv('submission.csv', index=False)
print(f"✓ Файл submission.csv сохранен ({len(df_result):,} записей)")

print("\n=== ВЫПОЛНЕНИЕ ЗАВЕРШЕНО ===")