import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import time

# ---------- Tabs Setup ----------
st.set_page_config(layout="wide")
st.title('Real-Time Market Forecasting using Meta Prophet by Vivek Bose')

tabs = st.tabs([
    "📈 Forecasting",
    "🧠 Model Insight",
    "📊 Raw Data",
    "📄 Project Overview",
    "Advanced Model Diagnostics"
])

# ---------- Shared Inputs ----------
with st.sidebar:
    ticker_symbol = st.text_input('Enter Ticker or Symbol (e.g., AAPL for Apple, RELIANCE.NS for RELIANCE):', 'AAPL')
    interval_options = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    selected_interval = st.selectbox('Select Time Interval:', interval_options, index=3)
    forecast_periods = st.slider('Forecast Periods (future points)', 1, 50, 5)
    run = st.button("▶️ Run Forecast")

# ---------- Data Fetch ----------
def fetch_data(ticker, interval):
    period = '1d' if interval.endswith('m') else '3mo'
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.index = df.index.tz_localize(None)
    return df

# ---------- Prophet ----------
def apply_prophet(df, periods, interval):
    df = df.reset_index().rename(columns={'Datetime': 'ds', 'Date': 'ds', 'Close': 'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    st.write("Recent Data")
    st.dataframe(df.tail(10))

    model = Prophet()
    model.fit(df)

    if interval.endswith('m'):
        freq = str(int(interval[:-1])) + 'T'
    elif interval.endswith('h'):
        freq = str(int(interval[:-1])) + 'H'
    else:
        freq = interval

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.tz_localize(None)
    return model, forecast, df

# ---------- Tab 1: Forecasting ----------
with tabs[0]:
    st.header("📈 Real-Time Forecasting")

    if run and ticker_symbol:
        try:
            df = fetch_data(ticker_symbol, selected_interval)
            model, forecast, df_prophet = apply_prophet(df, forecast_periods, selected_interval)

            fig, ax = plt.subplots()
            ax.plot(df.index, df['Close'], label='Original Price', color='blue')
            ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='red')
            ax.legend()
            ax.set_title(f"{ticker_symbol} Forecast")
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
            st.error(f"Error: {e}")
    else:
        st.info("Enter stock symbol and click ▶️ Run Forecast")

# ---------- Tab 2: Model Insight ----------
# with tabs[1]:
#     st.header("🧠 Prophet Model Components")
#     if run and 'model' in locals() and 'forecast' in locals():
#         fig2 = model.plot_components(forecast)
#         st.pyplot(fig2)
#     else:
#         st.info("Run a forecast first to view model insights.")


# ---------- Tab 2: Model Evaluation ----------
# with tabs[1]:
#     st.header("🧪 Model Evaluation")

#     if run and 'forecast' in locals() and 'df' in locals():
#         try:
#             # Align actuals and predictions
#             df_eval = df.copy()
#             df_eval = df_eval.reset_index().rename(columns={'Datetime': 'ds', 'Date': 'ds'})
#             df_eval['ds'] = df_eval['ds'].dt.tz_localize(None)

#             forecast_eval = forecast[['ds', 'yhat']]

#             # Merge on datetime
#             merged = pd.merge(df_eval, forecast_eval, on='ds', how='inner')
#             actual = merged['Close']
#             predicted = merged['yhat']

#             # Calculate metrics
#             mae = (actual - predicted).abs().mean()
#             rmse = ((actual - predicted) ** 2).mean() ** 0.5
#             mape = (abs((actual - predicted) / actual).mean()) * 100

#             # Display metrics
#             col1, col2, col3 = st.columns(3)
#             col1.metric("MAE", f"{mae:.2f}")
#             col2.metric("RMSE", f"{rmse:.2f}")
#             col3.metric("MAPE", f"{mape:.2f} %")

#             st.subheader("📉 Actual vs Predicted Comparison")
#             st.dataframe(merged[['ds', 'Close', 'yhat']].tail(20))

#             st.subheader("📈 Actual vs Predicted Plot")
#             fig3, ax3 = plt.subplots()
#             ax3.plot(merged['ds'], merged['Close'], label='Actual', color='blue')
#             ax3.plot(merged['ds'], merged['yhat'], label='Predicted', color='orange')
#             ax3.legend()
#             ax3.set_title('Actual vs Forecast')
#             ax3.tick_params(axis='x', rotation=45)
#             st.pyplot(fig3)

#         except Exception as e:
#             st.error(f"Evaluation Error: {e}")
#     else:
#         st.info("Run the forecast first to evaluate model.")

# ---------- Tab 2: Model Evaluation ----------
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

# ---------- Tab 4: Project Overview (Last Tab) ----------
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
    - 🧠 **Features**:
        - Intraday and daily forecasting
        - Prophet model decomposition (trend, seasonality)
        - Real-time input and updates

    ---
    **Experimentation by Vivek@Vivekbose.com**
    """)

# ---------- Tab 5: Advanced Model Diagnostics ----------
with tabs[4]:
    st.header("📉 Advanced Model Diagnostics")

    if run and 'forecast' in locals() and 'df' in locals() and 'model' in locals():
        try:
            # --- Actual vs Predicted Merging ---
            df_eval = df.copy()
            df_eval = df_eval.reset_index().rename(columns={'Datetime': 'ds', 'Date': 'ds'})
            df_eval['ds'] = df_eval['ds'].dt.tz_localize(None)
            forecast_eval = forecast[['ds', 'yhat']]
            merged = pd.merge(df_eval, forecast_eval, on='ds', how='inner')
            merged['residual'] = merged['Close'] - merged['yhat']

            # --- Residual Plot ---
            st.subheader("🔍 Residuals Over Time")
            fig5, ax5 = plt.subplots()
            ax5.plot(merged['ds'], merged['residual'], marker='o', linestyle='-', color='purple')
            ax5.axhline(0, color='gray', linestyle='--')
            ax5.set_title("Residual Plot (Actual - Predicted)")
            ax5.set_xlabel("Datetime")
            ax5.set_ylabel("Residual")
            ax5.tick_params(axis='x', rotation=45)
            st.pyplot(fig5)

            # --- Error Histogram ---
            st.subheader("📊 Forecast Error Distribution")
            fig6, ax6 = plt.subplots()
            ax6.hist(merged['residual'], bins=20, color='skyblue', edgecolor='black')
            ax6.set_title("Histogram of Forecast Errors")
            ax6.set_xlabel("Error (Close - yhat)")
            ax6.set_ylabel("Frequency")
            st.pyplot(fig6)

            # --- Seasonality Toggle ---
            st.subheader("⚙️ Prophet Seasonality Controls")

            # UI controls for user-customized Prophet model
            with st.form("seasonality_form"):
                daily = st.checkbox("Daily Seasonality", value=True)
                weekly = st.checkbox("Weekly Seasonality", value=True)
                yearly = st.checkbox("Yearly Seasonality", value=False)
                submit = st.form_submit_button("🔄 Retrain Model")

            if submit:
                # Retrain Prophet with custom seasonalities
                df_for_model = df.copy().reset_index().rename(columns={'Datetime': 'ds', 'Date': 'ds', 'Close': 'y'})
                df_for_model['ds'] = df_for_model['ds'].dt.tz_localize(None)
                custom_model = Prophet(
                    daily_seasonality=daily,
                    weekly_seasonality=weekly,
                    yearly_seasonality=yearly
                )
                custom_model.fit(df_for_model)

                future = custom_model.make_future_dataframe(periods=forecast_periods, freq='D')
                new_forecast = custom_model.predict(future)

                st.success("Model retrained with new seasonality settings!")
                fig7 = custom_model.plot_components(new_forecast)
                st.pyplot(fig7)

        except Exception as e:
            st.error(f"Diagnostics Error: {e}")
    else:
        st.info("Run the forecast first to access diagnostics.")


