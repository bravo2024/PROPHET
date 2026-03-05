import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

# ---------- Tabs Setup ----------
st.set_page_config("Real-Time Market Forecasting by Vivek Bose" ,layout="wide")
st.title('Real-Time Market Forecasting using Meta Prophet by Vivek Bose')

tabs = st.tabs([
    "📈 Forecasting",
    "🧪 Model Evaluation + Components",
    "📊 Raw Data",
    "📄 Project Overview",
    "⚙️ Model Persistence"
])

# ---------- Shared Inputs ----------
tickers_list = [
    'AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'FB', 'NFLX', 'NVDA', 'PYPL', 'INTC',
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'LT.NS', 'SBIN.NS',
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD',
    '^NSEI', '^NSEBANK', '^BSESN', '^DJI', '^GSPC', '^IXIC', '^FTSE', '^N225',
    'GC=F', 'CL=F', 'SI=F', 'NG=F', 'HG=F'
]

with st.sidebar:
    st.markdown("### 📈 Asset Symbol Options")

    # Text input + dropdown
    ticker_input = st.text_input("Type your ticker (overrides dropdown if filled):", "AAPL").upper()
    ticker_options = [ticker_input] + [t for t in tickers_list if t != ticker_input]
    ticker_symbol = st.selectbox("Or select any ticker from list:", ticker_options, index=0)

    # Interval dropdown
    interval_options = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
    selected_interval = st.selectbox("Select Time Interval:", interval_options, index=3)

    # Forecast periods
    forecast_periods = st.slider("Forecast Periods (future points)", 1, 50, 5)

    # Model persistence options
    model_save_name = st.text_input("Model save name (optional):", f"{ticker_symbol}_{selected_interval}")
    col1, col2 = st.columns(2)
    with col1:
        save_model = st.checkbox("💾 Save trained model", value=True)
    with col2:
        load_model = st.checkbox("📂 Load saved model", value=False)

    # Run button
    run = st.button("▶️ Run Forecast For Asset")

    # Display selections
    st.write("Ticker:", ticker_symbol)
    st.write("Interval:", selected_interval)
    st.write("Forecast Periods:", forecast_periods)

# ---------- Model Persistence Functions ----------
def save_prophet_model(model, filename):
    """Save Prophet model to disk"""
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/{filename}.pkl'
    joblib.dump(model, model_path)
    return model_path

def load_prophet_model(filename):
    """Load Prophet model from disk"""
    model_path = f'saved_models/{filename}.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ---------- Data Fetch with Error Handling ----------
def fetch_data(ticker, interval):
    try:
        period = '1d' if interval.endswith('m') else '3mo'
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            st.error(f"No data found for ticker '{ticker}' with interval '{interval}'")
            return None
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# ---------- Prophet with Model Caching ----------
def apply_prophet(df, periods, interval, model_name=None, load_existing=False):
    # Prepare data for Prophet
    df_prophet = df.reset_index().rename(columns={'Datetime': 'ds', 'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    # Check for existing model
    model = None
    if load_existing and model_name:
        model = load_prophet_model(model_name)
    
    # Train new model if not loaded
    if model is None:
        model = Prophet()
        model.fit(df_prophet)
        
        # Save if requested
        if save_model and model_name:
            model_path = save_prophet_model(model, model_name)
            st.sidebar.success(f"Model saved to: {model_path}")

    # Determine frequency for future dataframe
    if interval.endswith('m'):
        freq = str(int(interval[:-1])) + 'T'
    elif interval.endswith('h'):
        freq = str(int(interval[:-1])) + 'H'
    elif interval == '1wk':
        freq = 'W'
    elif interval == '1mo':
        freq = 'M'
    elif interval == '3mo':
        freq = '3M'
    else:
        freq = interval

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.tz_localize(None)
    
    return model, forecast, df_prophet

# ---------- Tab 1: Forecasting ----------
with tabs[0]:
    st.header("📈 Real-Time Forecasting")

    if run and ticker_symbol:
        df = fetch_data(ticker_symbol, selected_interval)
        if df is not None:
            try:
                model, forecast, df_prophet = apply_prophet(
                    df, forecast_periods, selected_interval, 
                    model_save_name, load_model
                )

                fig, ax = plt.subplots()
                ax.plot(df.index, df['Close'], label='Original Price', color='blue')
                ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='red')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                              alpha=0.2, color='red')
                ax.legend()
                ax.set_title(f"{ticker_symbol} Forecast with 80% Confidence Interval")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Actual Data")
                    st.dataframe(df.tail(20))
                with col2:
                    st.write("### Forecast Data")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods))
            except Exception as e:
                st.error(f"Model error: {e}")
    else:
        st.info("""Enter Stock/Crypto/Commodity Symbol from Yahoo Finance (e.g., US Stocks: `AAPL`, `MSFT`; Indian Stocks: `RELIANCE.NS`, `TCS.NS`; Crypto: `BTC-USD`, `ETH-USD`; Indices: `^NSEI`, `^DJI`; Commodities: `GC=F`, `CL=F`).  
Click ▶️ Run Forecast to see predictions (typing a ticker overrides dropdown selection).""")

# ---------- Tab 2: Model Evaluation + Components ----------
with tabs[1]:
    st.header("🧪 Model Evaluation + Prophet Components")

    if run and 'forecast' in locals() and 'df' in locals() and 'model' in locals():
        try:
            # Align actual and forecast
            df_eval = df.copy()
            df_eval = df_eval.reset_index().rename(columns={'Datetime': 'ds', 'Date': 'ds'})
            df_eval['ds'] = df_eval['ds'].dt.tz_localize(None)
            forecast_eval = forecast[['ds', 'yhat']]

            # Merge
            merged = pd.merge(df_eval, forecast_eval, on='ds', how='inner')
            actual = merged['Close']
            predicted = merged['yhat']

            # Evaluation Metrics
            mae = (actual - predicted).abs().mean()
            rmse = ((actual - predicted) ** 2).mean() ** 0.5
            mape = (abs((actual - predicted) / actual).mean()) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("MAPE", f"{mape:.2f} %")

            st.subheader("📉 Actual vs Predicted Table")
            st.dataframe(merged[['ds', 'Close', 'yhat']].tail(20))

            st.subheader("📈 Actual vs Forecast Plot")
            fig3, ax3 = plt.subplots()
            ax3.plot(merged['ds'], merged['Close'], label='Actual', color='blue')
            ax3.plot(merged['ds'], merged['yhat'], label='Predicted', color='orange')
            ax3.legend()
            ax3.set_title('Actual vs Forecast')
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)

            st.subheader("🧠 Prophet Model Components")
            fig4 = model.plot_components(forecast)
            st.pyplot(fig4)

        except Exception as e:
            st.error(f"Evaluation/Component Error: {e}")
    else:
        st.info("Run the forecast first to view evaluation and model internals.")

# ---------- Tab 3: Raw Data ----------
with tabs[2]:
    st.header("📊 Raw Stock Data")
    if run and 'df' in locals():
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(), file_name="stock_data.csv")
    else:
        st.info("No data available. Run a forecast to load data.")

# ---------- Tab 4: Project Overview ----------
with tabs[3]:
    st.header("📄 Project Overview")
    st.markdown("""
    ### Real-Time Stock Price Forecasting using Prophet
    
    - 🔍 **Objective**: Predict near-future stock prices using historical market data.
    - ⚙️ **Tech Stack**:
        - `Streamlit` for web interface
        - `Yahoo Finance API` (`yfinance`) for stock data
        - `Facebook Prophet` for time series forecasting
        - `Matplotlib` for plotting results
    - 🧠 **Advanced Features**:
        - **Model Persistence**: Save/load trained models to avoid retraining
        - **Error Handling**: Graceful handling of invalid tickers
        - **Multi-Asset Support**: Stocks, Crypto, Commodities, Indices
        - **Evaluation Metrics**: MAE, RMSE, MAPE with residual analysis

    ---
    **Experimentation by Vivek (GitHub: @bravo2024)**  
    **Live Deployment**: https://prophet.vivekailab.com/
    """)

# ---------- Tab 5: Model Persistence ----------
with tabs[4]:
    st.header("⚙️ Model Persistence Dashboard")
    
    # List saved models
    st.subheader("💾 Saved Models")
    if os.path.exists('saved_models'):
        models = os.listdir('saved_models')
        if models:
            for model_file in models:
                size_mb = os.path.getsize(f'saved_models/{model_file}') / (1024*1024)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"`{model_file}`")
                with col2:
                    st.write(f"{size_mb:.1f} MB")
                with col3:
                    if st.button(f"🗑️", key=f"del_{model_file}"):
                        os.remove(f'saved_models/{model_file}')
                        st.rerun()
        else:
            st.info("No saved models yet. Check 'Save trained model' in sidebar.")
    else:
        st.info("Saved models directory doesn't exist yet.")
    
    # Model comparison
    st.subheader("🔍 Model Performance Comparison")
    if st.button("Compare All Saved Models"):
        if os.path.exists('saved_models') and len(os.listdir('saved_models')) > 0:
            comparison_data = []
            for model_file in os.listdir('saved_models'):
                if model_file.endswith('.pkl'):
                    model_name = model_file.replace('.pkl', '')
                    size_mb = os.path.getsize(f'saved_models/{model_file}') / (1024*1024)
                    comparison_data.append({
                        'Model': model_name,
                        'Size (MB)': f"{size_mb:.2f}",
                        'Last Modified': os.path.getmtime(f'saved_models/{model_file}')
                    })
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
        else:
            st.warning("No saved models to compare.")