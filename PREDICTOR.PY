import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Function to load stock data
def load_data(symbol, timeframe, num_days=100):
    # Fetch historical stock data from Yahoo Finance for the last 100 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    df = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)

    # Reset index for compatibility with Prophet
    df.reset_index(inplace=True)

    return df

# Function to train Prophet model
def train_model(df):
    # Rename columns to 'ds' and 'y' for Prophet
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model
    model = Prophet()

    # Fit model to data
    model.fit(df)

    return model

# Function to make predictions with Prophet model
def predict(model, future):
    # Make predictions for the future period
    forecast = model.predict(future)

    return forecast

# Function to display results
def display_results(df, forecast):
    # Plot actual vs. predicted closing prices
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot actual closing prices
    ax.plot(df['ds'], df['y'], label='Actual')

    # Plot predicted closing prices
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted')

    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Actual vs. Predicted Closing Prices')
    ax.legend()

    st.pyplot(fig)

def main():
    st.title("Live Stock Analysis")

    # User input for stock symbol and timeframe
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple Inc.)")
    timeframe = st.selectbox("Select Timeframe", ['1d', '1wk', '1mo'])

    if symbol:
        # Load data
        df = load_data(symbol, timeframe)

        if not df.empty:
            # Train Prophet model
            model = train_model(df)

            # Make future dataframe for forecasting
            future = model.make_future_dataframe(periods=30)  # Forecast next 30 days

            # Make predictions
            forecast = predict(model, future)

            # Display results in columns
            col1, col2 = st.beta_columns(2)
            with col1:
                st.write("### Actual Data")
                st.write(df)
            with col2:
                st.write("### Forecast Data")
                st.write(forecast)

            # Plot results
            st.write("### Plot")
            display_results(df, forecast)

if __name__ == "__main__":
    main()
