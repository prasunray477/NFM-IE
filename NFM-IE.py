import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from plotly import graph_objs as go
import warnings
from load_dataset import load_data
from neutrosophic import *

warnings.filterwarnings('ignore')


# Page configurations: ICON, Title
st.set_page_config(page_title='NFM-IE', layout='centered', page_icon='📈')
st.title('NFM-IE: Neutrosophic Fluctuation Model for Stock Price Forecasting')


# Sidebar for user input
stock_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOG)",value="AAPL" ).upper()

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2025, 1, 1),
    min_value=datetime(1900, 1, 1),
    max_value=date.today()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=start_date,
    max_value=date.today()
)

# Button to fetch data
if st.sidebar.button('Fetch Data'):
    if not stock_input:
        st.error('Please enter a stock ticker!')
    elif start_date >= end_date:
        st.error('Start date must be before end date!')
    else:
        with st.spinner(f'Loading data for {stock_input}...'):
            data = load_data(stock_input, start_date, end_date)
        
        if data is None or data.empty:
            st.error(f'No data found for ticker: {stock_input}. Please check the ticker symbol.')
        else:
            st.success(f' {stock_input} stock Data has been successfully loaded from Yahoo Finance!')
            
            # Display raw data 
            with st.expander("Raw Data", expanded=True):
                st.dataframe(data.head(), width='stretch')
            
            # Data Visualization
            with st.expander("Data Visualization", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name="Open Price"))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name="Close Price"))        
                fig.update_layout(
                    title=f'{stock_input} Stock Price Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    xaxis_rangeslider_visible=True,
                    hovermode='x unified',
                    height=600
                )
                st.plotly_chart(fig, width='stretch')
            
            # Close Price Fluctuation
            with st.expander("Close Price Fluctuation", expanded=True):
                data['Daily_Change'] = data['Close'].diff()
                data['Daily_Change_Pct'] = data['Close'].pct_change() * 100
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=data['Date'], 
                    y=data['Daily_Change_Pct'],
                    marker_color=['red' if val < 0 else 'green' for val in data['Daily_Change_Pct']],
                    name="Daily % Change"
                ))
                fig2.update_layout(
                    title=f'{stock_input} Daily Price Fluctuation (%)',
                    xaxis_title='Date',
                    yaxis_title='Change (%)',
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig2, width='stretch')


# NFM-IE Model Training & Forecasting
with st.expander("Model Training & Forecasting", expanded=True):
    st.subheader("Train NFM-IE Model and Forecast")
    if 'data' in locals() and not data.empty:
        V_train = data['Close'].values
        model = NFM_IE(m=9, threshold=0.89)
        with st.spinner('Training model...'):
            model.fit(V_train[:-30])  # Use all but last 30 days for training
        st.success('Model trained successfully!')

        #amount of data trained
        st.write(f"Data used to train this model: {len(V_train[:-30])} days")

        # Map neutrosophic points for visualization
        T_vals, I_vals, F_vals = map_neutrosophic_points(model.NFTS_train)
        st.write("Neutrosophic Mapping:")
        st.scatter_chart(pd.DataFrame({'Truth': T_vals, 'Indeterminacy': I_vals, 'Falsity': F_vals}))

        with st.spinner('Forecasting next 30 days...'):
            forecast = model.predict(V_train[:-30], steps=30)
            st.success('Forecasting completed!')
        
        # Display forecasted values
        forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Close': forecast})
        st.dataframe(forecast_df)
        
        # Calculate metrics against actual last 30 days
        actual = V_train[-30:]
        
        # Plot next 30 days forecast vs actual
        fig3 = go.Figure()
        actual_dates = data['Date'].iloc[-30:].values
        fig3.add_trace(go.Scatter(x=actual_dates, y=actual, mode='lines', name='Actual', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
        fig3.update_layout(
            title=f'{stock_input} Actual vs Forecast Close Price (for next 30 days)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig3, width='stretch')


        # Calculate and display metrics
        metrics = calculate_metrics(actual, forecast)
        st.json(metrics)

