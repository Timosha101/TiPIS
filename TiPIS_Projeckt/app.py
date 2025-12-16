# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Прогнозирование цен на криптовалюты",
    page_icon="📈",
    layout="wide"
)

# Заголовок
st.title("📈 Прогнозирование цен на криптовалюты")
st.markdown("""
Это приложение использует алгоритмы для прогнозирования цен на криптовалюты.
Выберите криптовалюту и параметры для получения прогноза.
""")

# Сайдбар для настроек
st.sidebar.header("⚙️ Настройки")

# Список доступных криптовалют (используем yfinance напрямую)
CRYPTO_LIST = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD',
    'SHIB-USD', 'TRX-USD', 'LINK-USD', 'UNI-USD', 'ATOM-USD'
]

# Загрузка данных с yfinance
@st.cache_data(ttl=3600)  # Кэшируем на 1 час
def load_crypto_data(crypto_symbol, period='6mo'):
    """Загрузка данных криптовалюты с yfinance"""
    try:
        ticker = yf.Ticker(crypto_symbol)
        df = ticker.history(period=period)
        if df.empty:
            # Пробуем альтернативный период
            df = ticker.history(period='1y')
        
        if df.empty:
            st.error(f"Не удалось загрузить данные для {crypto_symbol}")
            return None
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных для {crypto_symbol}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Расчет RSI"""
    if len(prices) < period + 1:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(prices, window):
    """Расчет простого скользящего среднего"""
    if len(prices) < window:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    return prices.rolling(window=window).mean()

def generate_forecast(crypto_data, forecast_days):
    """Генерация прогноза на основе исторических данных"""
    if crypto_data is None or len(crypto_data) < 10:
        return None, None, None
    
    # Используем несколько методов для прогнозирования
    prices = crypto_data['Close'].values
    
    # Метод 1: Расчет доходности - ИСПРАВЛЕННЫЙ КОД
    if len(prices) >= 2:
        # Правильный расчет доходности
        returns = np.diff(prices) / prices[:-1]
        
        # Используем последние 30 значений или все, если меньше
        lookback = min(30, len(returns))
        recent_returns = returns[-lookback:] if len(returns) > 0 else []
        
        if len(recent_returns) > 0:
            avg_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
        else:
            avg_return = 0
            std_return = 0.02  # Дефолтная волатильность 2%
    else:
        avg_return = 0
        std_return = 0.02
    
    # Метод 2: Тренд на основе скользящих средних
    if len(prices) >= 20:
        sma_short = calculate_sma(pd.Series(prices), min(10, len(prices)))
        sma_long = calculate_sma(pd.Series(prices), min(20, len(prices)))
        
        if not sma_short.isna().iloc[-1] and not sma_long.isna().iloc[-1]:
            trend_factor = 1.0 + (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1] * 0.3
        else:
            trend_factor = 1.0
    else:
        trend_factor = 1.0  # Нейтральный тренд при недостатке данных
    
    # Генерация прогноза
    forecast_prices = []
    last_price = prices[-1]
    
    for i in range(forecast_days):
        # Комбинируем тренд и случайные колебания
        random_factor = np.random.normal(avg_return, std_return * 0.7)
        forecast_price = last_price * (1 + random_factor) * trend_factor
        
        # Ограничиваем экстремальные значения
        max_change = 0.15  # Максимальное дневное изменение 15%
        change = (forecast_price - last_price) / last_price
        if abs(change) > max_change:
            forecast_price = last_price * (1 + np.sign(change) * max_change)
            
        forecast_prices.append(forecast_price)
        last_price = forecast_price
    
    # Сглаживаем прогноз
    if len(forecast_prices) >= 3:
        forecast_prices = pd.Series(forecast_prices).rolling(
            window=min(3, len(forecast_prices)), 
            center=True, 
            min_periods=1
        ).mean().tolist()
    
    # Генерация дат прогноза
    last_date = crypto_data['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    return forecast_dates, forecast_prices, prices

# Основной интерфейс
def main():
    # Выбор криптовалюты
    selected_crypto = st.sidebar.selectbox(
        "Выберите криптовалюту:",
        CRYPTO_LIST,
        index=0
    )
    
    # Выбор периода данных
    data_period = st.sidebar.selectbox(
        "Период исторических данных:",
        ['1mo', '3mo', '6mo', '1y', '2y'],
        index=2
    )
    
    # Параметры прогноза
    forecast_days = st.sidebar.slider(
        "Дней для прогноза:",
        min_value=1,
        max_value=30,
        value=7,
        step=1
    )
    
    # Загрузка данных
    with st.spinner(f"Загрузка данных для {selected_crypto}..."):
        crypto_data = load_crypto_data(selected_crypto, data_period)
    
    if crypto_data is not None and len(crypto_data) > 0:
        # Основная область
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(f"Анализ: {selected_crypto}")
            
            # График цен
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=crypto_data['Date'],
                y=crypto_data['Close'],
                mode='lines',
                name='Цена закрытия',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Дата: %{x}<br>Цена: $%{y:,.2f}<extra></extra>'
            ))
            
            # Добавляем скользящие средние если данных достаточно
            if len(crypto_data) > 20:
                crypto_data['SMA_20'] = calculate_sma(crypto_data['Close'], 20)
                fig.add_trace(go.Scatter(
                    x=crypto_data['Date'],
                    y=crypto_data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#ff7f0e', width=1.5, dash='dash'),
                    opacity=0.7
                ))
            
            fig.update_layout(
                title=f'Исторические цены {selected_crypto}',
                xaxis_title='Дата',
                yaxis_title='Цена (USD)',
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Показатели
            st.subheader("📊 Ключевые показатели")
            col1_1, col1_2, col1_3, col1_4 = st.columns(4)
            
            with col1_1:
                current_price = crypto_data['Close'].iloc[-1]
                st.metric(
                    "Текущая цена", 
                    f"${current_price:,.2f}",
                    delta=None
                )
            
            with col1_2:
                if len(crypto_data) > 1:
                    daily_change = ((crypto_data['Close'].iloc[-1] - crypto_data['Close'].iloc[-2]) / 
                                   crypto_data['Close'].iloc[-2] * 100)
                    st.metric(
                        "Изменение за день", 
                        f"{daily_change:+.2f}%",
                        delta=f"{daily_change:+.2f}%"
                    )
                else:
                    st.metric("Изменение за день", "N/A")
            
            with col1_3:
                if 'Volume' in crypto_data.columns:
                    volume = crypto_data['Volume'].iloc[-1]
                    # Форматируем объем
                    if volume > 1e9:
                        vol_text = f"${volume/1e9:.2f}B"
                    elif volume > 1e6:
                        vol_text = f"${volume/1e6:.2f}M"
                    else:
                        vol_text = f"${volume:,.0f}"
                    st.metric("Объем", vol_text)
            
            with col1_4:
                if len(crypto_data) >= 30:
                    volatility = crypto_data['Close'].pct_change().std() * np.sqrt(365) * 100
                    st.metric("Годовая волатильность", f"{volatility:.1f}%")
        
        with col2:
            st.header("Прогноз")
            
            # Кнопка для прогнозирования
            if st.button("🔄 Сгенерировать прогноз", type="primary", use_container_width=True):
                with st.spinner("Выполняется прогнозирование..."):
                    # Генерация прогноза
                    forecast_dates, forecast_prices, historical_prices = generate_forecast(
                        crypto_data, forecast_days
                    )
                    
                    if forecast_prices is not None:
                        # График прогноза
                        fig_forecast = go.Figure()
                        
                        # Исторические данные (последние 60 дней или все, если меньше)
                        history_days = min(60, len(crypto_data))
                        fig_forecast.add_trace(go.Scatter(
                            x=crypto_data['Date'].iloc[-history_days:],
                            y=crypto_data['Close'].iloc[-history_days:],
                            mode='lines',
                            name='История',
                            line=dict(color='#1f77b4', width=2),
                            hovertemplate='Дата: %{x}<br>Цена: $%{y:,.2f}<extra></extra>'
                        ))
                        
                        # Прогноз
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_prices,
                            mode='lines+markers',
                            name='Прогноз',
                            line=dict(color='#d62728', width=2, dash='dash'),
                            marker=dict(size=6, color='#d62728'),
                            hovertemplate='Дата: %{x}<br>Прогноз: $%{y:,.2f}<extra></extra>'
                        ))
                        
                        fig_forecast.update_layout(
                            title=f'Прогноз цен на {forecast_days} дней',
                            xaxis_title='Дата',
                            yaxis_title='Цена (USD)',
                            height=400,
                            template='plotly_white',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Показатели прогноза
                        st.subheader("📈 Метрики прогноза")
                        
                        last_price = crypto_data['Close'].iloc[-1]
                        forecast_end_price = forecast_prices[-1]
                        price_change = ((forecast_end_price - last_price) / last_price * 100)
                        
                        col2_1, col2_2, col2_3 = st.columns(3)
                        with col2_1:
                            st.metric(
                                f"Через {forecast_days} дней",
                                f"${forecast_end_price:,.2f}",
                                f"{price_change:+.2f}%",
                                delta_color="normal"
                            )
                        
                        with col2_2:
                            avg_forecast = np.mean(forecast_prices)
                            avg_change = ((avg_forecast - last_price) / last_price * 100)
                            st.metric(
                                "Средний прогноз",
                                f"${avg_forecast:,.2f}",
                                f"{avg_change:+.2f}%"
                            )
                        
                        with col2_3:
                            volatility = np.std(forecast_prices) / avg_forecast * 100
                            st.metric("Волатильность прогноза", f"{volatility:.1f}%")
        
        # Нижняя часть - технический анализ
        st.header("📊 Технический анализ")
        
        if len(crypto_data) > 20:
            # Расчет индикаторов
            crypto_data['SMA_20'] = calculate_sma(crypto_data['Close'], 20)
            crypto_data['RSI'] = calculate_rsi(crypto_data['Close'])
            
            # График с индикаторами
            fig_indicators = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Цена и скользящие средние', 'RSI (14 периодов)'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True
            )
            
            # Цена и SMA
            fig_indicators.add_trace(
                go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'],
                          name='Цена закрытия', line=dict(color='#1f77b4', width=2)),
                row=1, col=1
            )
            
            if not crypto_data['SMA_20'].isna().all():
                fig_indicators.add_trace(
                    go.Scatter(x=crypto_data['Date'], y=crypto_data['SMA_20'],
                              name='SMA 20', line=dict(color='#ff7f0e', width=1.5)),
                    row=1, col=1
                )
            
            # RSI
            if not crypto_data['RSI'].isna().all():
                fig_indicators.add_trace(
                    go.Scatter(x=crypto_data['Date'], y=crypto_data['RSI'],
                              name='RSI', line=dict(color='#9467bd', width=2)),
                    row=2, col=1
                )
                
                # Уровни RSI
                fig_indicators.add_hline(
                    y=70, line_dash="dash", line_color="red", 
                    annotation_text="Перекупленность", annotation_position="bottom right",
                    row=2, col=1
                )
                fig_indicators.add_hline(
                    y=30, line_dash="dash", line_color="green", 
                    annotation_text="Перепроданность", annotation_position="top right",
                    row=2, col=1
                )
                fig_indicators.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
            
            fig_indicators.update_layout(
                height=600, 
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Обновляем оси
            fig_indicators.update_yaxes(title_text="Цена (USD)", row=1, col=1)
            fig_indicators.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig_indicators.update_xaxes(title_text="Дата", row=2, col=1)
            
            st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Интерпретация RSI
            if not crypto_data['RSI'].isna().iloc[-1]:
                current_rsi = crypto_data['RSI'].iloc[-1]
                if not np.isnan(current_rsi):
                    st.subheader("📈 Интерпретация RSI")
                    if current_rsi > 70:
                        st.warning(f"RSI: {current_rsi:.1f} - Сигнал перекупленности. Возможна коррекция.")
                    elif current_rsi < 30:
                        st.success(f"RSI: {current_rsi:.1f} - Сигнал перепроданности. Возможен рост.")
                    else:
                        st.info(f"RSI: {current_rsi:.1f} - Нейтральная зона.")
        else:
            st.info("Для технического анализа требуется более 20 дней данных. Выберите более длительный период.")
    
    else:
        st.error("Не удалось загрузить данные. Пожалуйста, проверьте подключение к интернету и попробуйте другую криптовалюту.")

if __name__ == "__main__":
    main()