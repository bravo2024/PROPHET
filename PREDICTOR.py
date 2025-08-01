import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to load stock data
# def load_data(symbol, timeframe, num_days=44):
#     # Fetch historical stock data from Yahoo Finance for the last 100 days
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=num_days)
#     df = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)

#     # Reset index for compatibility with Prophet
#     df.reset_index(inplace=True)

#     return df


def load_data(symbol, timeframe, num_days=44):
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)

    # Download stock data from Yahoo Finance
    df = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)

    # Flatten MultiIndex columns if present (e.g., df.columns = [('Close', 'RELIANCE.NS'), ...])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to move Date/Datetime into a column
    df.reset_index(inplace=True)

    # Validate required columns
    if 'Close' not in df.columns:
        raise ValueError("Missing 'Close' column in the Yahoo Finance data.")
    if not any(col in df.columns for col in ['Date', 'Datetime']):
        raise ValueError("Missing 'Date' or 'Datetime' column in the data.")

    # Optional: Debug info (uncomment in Streamlit)
    # import streamlit as st
    # st.write("Yahoo Finance Data:")
    # st.dataframe(df.head())
    # st.write("Columns:", df.columns.tolist())
    # st.write("Data Types:", df.dtypes)

    return df

# Function to train Prophet model
def train_model(df):
    # Rename columns to 'ds' and 'y' for Prophet
    #df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    df['ds'] = df['ds'].dt.tz_localize(None)
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

    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ds'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds'})
    
    forecast=forecast[:-1]
    #fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size
    #ax.plot(df['ds'], df['Close'], label='Actual', color='blue', linestyle='-')
    #ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='red', linestyle='--')
    #ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)  # Fill confidence interval
    #ax.set_xlabel('Date')
    #ax.set_ylabel('Closing Price')
    #ax.set_title('Actual vs. Predicted Closing Prices')
    #ax.legend()
    #plt.xticks(rotation=45)
    #ax.autoscale(enable=True, axis='y', tight=True)  # Autoscale y-axis
    #plt.tight_layout()  # Adjust layout
    # Rotate x-axis labels for better readability
    #plt.tight_layout()  # Adjust layout
    #st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot actual closing prices
    ax.plot(df['ds'], df['Close'], label='Actual', color='blue')

    # Plot forecasted closing prices with confidence interval
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Actual vs. Forecasted Closing Prices')
    ax.legend()
    plt.xticks(rotation=45)

    # Show plot
    plt.tight_layout()
    st.pyplot(fig)




def main():
    st.title("Live Stock Analysis")

    # User input for stock symbol and timeframe
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple Inc.)")
    timeframe = st.selectbox("Select Timeframe", ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'])

    if symbol:
        df = load_data(symbol, timeframe)
        if not df.empty:
            model = train_model(df)
            forecast_periods = st.slider("Select Number of Forecast Periods", min_value=1, max_value=365, value=3)
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = predict(model, future)
           

            # Display results in columns
            col1, col2 = st.columns(2)
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
