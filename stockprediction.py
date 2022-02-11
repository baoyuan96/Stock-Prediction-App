##Stock Prediction
#commands to run file
#pip install yfinance prophet plotly
#streamlit run stockprediction.py

import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Python")

stocks = ("AAPL", "GOOGL", "MSFT", "TSLA", "NFLX", "AMZN", "FB") #7 stocks for now
selected_stock = st.selectbox("Select dataset for prediction", stocks) #dropdown slectbox to choose stocks

num_years = st.slider("Years of prediction:", 1, 4)
period = num_years * 365  #period of days


@st.cache #cache the yfinance download data so not req to redownload selected stocfks
def load_data(ticker):
    data = yf.download(ticker,START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

st.subheader('Raw data')
st.write(data.tail())

#Plotting raw data table 
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Predicting with Prophet 
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#Show and plot prediction
st.subheader('Prediction data')
st.write(forecast.tail())

st.write(f'Prediction plot for {num_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Prediction components")
fig2 = m.plot_components(forecast)
st.write(fig2)
