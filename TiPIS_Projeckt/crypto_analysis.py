# crypto_analysis.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Машинное обучение
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Глубокое обучение
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Загрузка данных
def load_crypto_data(filepath='data/Crypto_historical_data.csv'):
    """Загрузка и предварительная обработка данных"""
    df = pd.read_csv(filepath)
    
    # Преобразование даты
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(['Cryptocurrency', 'Date'], inplace=True)
    
    # Проверка пропущенных значений
    print("Информация о данных:")
    print(df.info())
    print(f"\nРазмер датасета: {df.shape}")
    print(f"\nКоличество криптовалют: {df['Cryptocurrency'].nunique()}")
    print(f"\nДиапазон дат: {df['Date'].min()} до {df['Date'].max()}")
    
    return df

# Загрузка данных
df = load_crypto_data()
def analyze_and_prepare_data(df, crypto_name='Bitcoin'):
    """Анализ и подготовка данных для конкретной криптовалюты"""
    
    # Фильтрация данных по выбранной криптовалюте
    crypto_df = df[df['Cryptocurrency'] == crypto_name].copy()
    
    # Проверка пропущенных значений
    print(f"\nАнализ данных для {crypto_name}:")
    print(f"Количество записей: {len(crypto_df)}")
    print(f"Пропущенные значения:")
    print(crypto_df.isnull().sum())
    
    # Заполнение пропусков (если есть)
    crypto_df['Close'].fillna(method='ffill', inplace=True)
    crypto_df['Volume'].fillna(0, inplace=True)
    
    # Создание временных признаков
    crypto_df['Year'] = crypto_df['Date'].dt.year
    crypto_df['Month'] = crypto_df['Date'].dt.month
    crypto_df['Day'] = crypto_df['Date'].dt.day
    crypto_df['Day_of_week'] = crypto_df['Date'].dt.dayofweek
    crypto_df['Quarter'] = crypto_df['Date'].dt.quarter
    
    # Технические индикаторы
    crypto_df['SMA_7'] = crypto_df['Close'].rolling(window=7).mean()
    crypto_df['SMA_30'] = crypto_df['Close'].rolling(window=30).mean()
    crypto_df['EMA_12'] = crypto_df['Close'].ewm(span=12).mean()
    crypto_df['EMA_26'] = crypto_df['Close'].ewm(span=26).mean()
    
    # MACD
    crypto_df['MACD'] = crypto_df['EMA_12'] - crypto_df['EMA_26']
    crypto_df['MACD_signal'] = crypto_df['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = crypto_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    crypto_df['RSI'] = 100 - (100 / (1 + rs))
    
    # Волатильность
    crypto_df['Volatility'] = crypto_df['Close'].rolling(window=20).std()
    
    # Целевая переменная - цена на следующий день
    crypto_df['Target'] = crypto_df['Close'].shift(-1)
    
    # Удаление строк с NaN после создания индикаторов
    crypto_df.dropna(inplace=True)
    
    return crypto_df

# Анализ Bitcoin
btc_df = analyze_and_prepare_data(df, 'Bitcoin')
def prepare_features_for_modeling(crypto_df, test_size=0.2, validation_size=0.1):
    """Подготовка признаков для моделей машинного обучения"""
    
    # Выбор признаков
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_signal', 'RSI', 'Volatility',
        'Year', 'Month', 'Day', 'Day_of_week', 'Quarter'
    ]
    
    X = crypto_df[feature_columns].values
    y = crypto_df['Target'].values
    
    # Разбиение данных с сохранением временного порядка
    n_samples = len(X)
    n_train = int(n_samples * (1 - test_size - validation_size))
    n_val = int(n_samples * validation_size)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    # Масштабирование признаков
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    print(f"Размеры данных:")
    print(f"Обучающая выборка: {X_train_scaled.shape}")
    print(f"Валидационная выборка: {X_val_scaled.shape}")
    print(f"Тестовая выборка: {X_test_scaled.shape}")
    
    return (X_train_scaled, y_train_scaled, 
            X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test,
            scaler_y, feature_columns)

# Подготовка данных для моделирования
(X_train, y_train, X_val, y_val, 
 X_test, y_test, scaler_y, feature_columns) = prepare_features_for_modeling(btc_df)
def train_ml_models(X_train, y_train, X_val, y_val):
    """Обучение нескольких традиционных моделей ML"""
    
    models = {}
    results = {}
    
    # 1. Random Forest
    print("Обучение Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # 2. Gradient Boosting
    print("Обучение Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models['GradientBoosting'] = gb_model
    
    # 3. XGBoost
    print("Обучение XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # Оценка моделей на валидационной выборке
    for name, model in models.items():
        y_pred_val = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"\n{name} результаты:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    return models, results

# Обучение ML моделей
ml_models, ml_results = train_ml_models(X_train, y_train, X_val, y_val)
def create_lstm_model(input_shape):
    """Создание LSTM модели для прогнозирования временных рядов"""
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_sequences(data, targets, seq_length=30):
    """Создание последовательностей для LSTM"""
    X_seq, y_seq = [], []
    
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(targets[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)

# Подготовка данных для LSTM
seq_length = 30
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

print(f"LSTM последовательности:")
print(f"Обучающие: {X_train_seq.shape}, {y_train_seq.shape}")
print(f"Валидационные: {X_val_seq.shape}, {y_val_seq.shape}")

# Создание и обучение LSTM модели
lstm_model = create_lstm_model((seq_length, X_train_seq.shape[2]))

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'models/best_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True
)

print("\nОбучение LSTM модели...")
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Оценка LSTM модели
lstm_pred_val = lstm_model.predict(X_val_seq)
lstm_mse = mean_squared_error(y_val_seq, lstm_pred_val)
lstm_mae = mean_absolute_error(y_val_seq, lstm_pred_val)
lstm_r2 = r2_score(y_val_seq, lstm_pred_val)

print(f"\nLSTM результаты:")
print(f"MSE: {lstm_mse:.4f}")
print(f"MAE: {lstm_mae:.4f}")
print(f"R² Score: {lstm_r2:.4f}")
def create_cnn_lstm_model(input_shape):
    """Создание гибридной CNN-LSTM модели"""
    
    inputs = Input(shape=input_shape)
    
    # CNN слои для извлечения признаков
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM слои для временных зависимостей
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    
    # Полносвязные слои
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Создание и обучение CNN-LSTM модели
cnn_lstm_model = create_cnn_lstm_model((seq_length, X_train_seq.shape[2]))

print("\nОбучение CNN-LSTM модели...")
cnn_lstm_history = cnn_lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Оценка CNN-LSTM модели
cnn_lstm_pred_val = cnn_lstm_model.predict(X_val_seq)
cnn_lstm_mse = mean_squared_error(y_val_seq, cnn_lstm_pred_val)
cnn_lstm_mae = mean_absolute_error(y_val_seq, cnn_lstm_pred_val)
cnn_lstm_r2 = r2_score(y_val_seq, cnn_lstm_pred_val)

print(f"\nCNN-LSTM результаты:")
print(f"MSE: {cnn_lstm_mse:.4f}")
print(f"MAE: {cnn_lstm_mae:.4f}")
print(f"R² Score: {cnn_lstm_r2:.4f}")
def compare_models(results_dict, lstm_results, cnn_lstm_results):
    """Сравнение всех моделей и выбор лучшей"""
    
    # Сбор всех результатов
    all_results = {
        'RandomForest': results_dict['RandomForest'],
        'GradientBoosting': results_dict['GradientBoosting'],
        'XGBoost': results_dict['XGBoost'],
        'LSTM': {
            'mse': lstm_results[0],
            'mae': lstm_results[1],
            'r2': lstm_results[2]
        },
        'CNN-LSTM': {
            'mse': cnn_lstm_results[0],
            'mae': cnn_lstm_results[1],
            'r2': cnn_lstm_results[2]
        }
    }
    
    # Создание DataFrame для сравнения
    comparison_df = pd.DataFrame({
        'Model': list(all_results.keys()),
        'MSE': [all_results[m]['mse'] for m in all_results.keys()],
        'MAE': [all_results[m]['mae'] for m in all_results.keys()],
        'R2': [all_results[m]['r2'] for m in all_results.keys()]
    })
    
    print("\nСравнение моделей:")
    print(comparison_df)
    
    # Визуализация сравнения
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(comparison_df['Model'], comparison_df['MSE'])
    axes[0].set_title('Сравнение MSE')
    axes[0].set_ylabel('MSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(comparison_df['Model'], comparison_df['MAE'])
    axes[1].set_title('Сравнение MAE')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(comparison_df['Model'], comparison_df['R2'])
    axes[2].set_title('Сравнение R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Выбор лучшей модели по R² Score
    best_model_name = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
    best_model_score = comparison_df.loc[comparison_df['R2'].idxmax(), 'R2']
    
    print(f"\nЛучшая модель: {best_model_name} с R² = {best_model_score:.4f}")
    
    return best_model_name, comparison_df

# Сравнение всех моделей
lstm_results = (lstm_mse, lstm_mae, lstm_r2)
cnn_lstm_results = (cnn_lstm_mse, cnn_lstm_mae, cnn_lstm_r2)

best_model, comparison_df = compare_models(ml_results, lstm_results, cnn_lstm_results)
def test_best_model(best_model_name, models_dict, 
                   X_test_seq, y_test_seq, X_test, y_test, 
                   scaler_y, seq_length=30):
    """Тестирование лучшей модели на тестовой выборке"""
    
    if 'LSTM' in best_model_name:
        # Для LSTM моделей
        if best_model_name == 'LSTM':
            model = lstm_model
            predictions = model.predict(X_test_seq)
        else:
            model = cnn_lstm_model
            predictions = model.predict(X_test_seq)
        
        # Преобразование обратно к оригинальному масштабу
        predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        
    else:
        # Для традиционных ML моделей
        model = models_dict[best_model_name]['model']
        predictions_scaled = model.predict(X_test)
        
        # Преобразование обратно к оригинальному масштабу
        predictions_original = scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        y_test_original = scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
    
    # Расчет метрик
    mse_test = mean_squared_error(y_test_original, predictions_original)
    mae_test = mean_absolute_error(y_test_original, predictions_original)
    r2_test = r2_score(y_test_original, predictions_original)
    
    print(f"\nТестирование лучшей модели ({best_model_name}):")
    print(f"MSE на тесте: {mse_test:.4f}")
    print(f"MAE на тесте: {mae_test:.4f}")
    print(f"R² Score на тесте: {r2_test:.4f}")
    
    # Визуализация предсказаний
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original[:100], label='Фактические значения', alpha=0.7)
    plt.plot(predictions_original[:100], label='Предсказания', alpha=0.7)
    plt.title(f'Предсказания vs Фактические значения ({best_model_name})')
    plt.xlabel('Временные точки')
    plt.ylabel('Цена (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('best_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return predictions_original, y_test_original

# Тестирование лучшей модели
predictions, actual = test_best_model(
    best_model, ml_results, 
    X_test_seq, y_test_seq, X_test, y_test, 
    scaler_y, seq_length
)