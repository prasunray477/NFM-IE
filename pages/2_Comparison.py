# Compare Models Page
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from plotly import graph_objs as go
from load_dataset import load_data
from neutrosophic import *
from arima import arima_forecast, calculate_metrics as arima_calc_metrics
from lstm import lstm_forecast, calculate_metrics as lstm_calc_metrics


st.title('Model Comparison: NFM-IE vs ARIMA vs LSTM')

stock_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="AAPL").upper()

start_date = st.date_input(
    "Start Date",
    value=datetime(2025, 1, 1),
    min_value=datetime(1900, 1, 1),
    max_value=date.today()
)

end_date = st.date_input(
    "End Date",
    value=date.today(),
    min_value=start_date,
    max_value=date.today()
)

if st.button('Run Comparison'):
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
            st.success(f'Data loaded successfully for {stock_input}!')
            
            with st.spinner('Running model comparison...'):
                V_train = data['Close'].values
                actual = V_train[-30:]
                train_data = V_train[:-30]
                
                # NFM-IE
                nfm_model = NFM_IE(m=9, threshold=0.89)
                nfm_model.fit(train_data)
                nfm_pred = nfm_model.predict(train_data, steps=30)
                nfm_metrics = calculate_metrics(actual, nfm_pred)
                
                # ARIMA
                try:
                    arima_pred = arima_forecast(stock_input, start_date, end_date)
                    arima_metrics = calculate_metrics(actual, arima_pred) if arima_pred is not None else None
                except Exception as e:
                    arima_pred = None
                    arima_metrics = None
                    st.warning(f'ARIMA failed: {str(e)}')
                
                # LSTM
                try:
                    lstm_pred = lstm_forecast(stock_input, start_date, end_date)
                    lstm_metrics = lstm_calc_metrics(actual, lstm_pred) if lstm_pred is not None else None
                except Exception as e:
                    lstm_pred = None
                    lstm_metrics = None
                    st.warning(f'LSTM failed: {str(e)}')
                
                st.success('Comparison completed!')
                
                # Plot comparison
                fig_comp = go.Figure()
                actual_dates = data['Date'].iloc[-30:].values
                fig_comp.add_trace(go.Scatter(x=actual_dates, y=actual, mode='lines', name='Actual', line=dict(color='blue', width=3)))
                fig_comp.add_trace(go.Scatter(x=actual_dates, y=nfm_pred, mode='lines', name='NFM-IE', line=dict(color='green', dash='dash')))
                if arima_pred is not None:
                    fig_comp.add_trace(go.Scatter(x=actual_dates, y=arima_pred, mode='lines', name='ARIMA', line=dict(color='orange', dash='dot')))
                if lstm_pred is not None:
                    fig_comp.add_trace(go.Scatter(x=actual_dates, y=lstm_pred, mode='lines', name='LSTM', line=dict(color='red', dash='dashdot')))
                fig_comp.update_layout(
                    title='Model Comparison: Forecasts vs Actual',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    height=600
                )
                st.plotly_chart(fig_comp, width='stretch')
                
                # Metrics comparison
                metrics_data = {'NFM-IE': nfm_metrics}
                if arima_metrics:
                    metrics_data['ARIMA'] = arima_metrics
                if lstm_metrics:
                    metrics_data['LSTM'] = lstm_metrics
                
                fig_metrics = go.Figure()
                for metric in ['RMSE', 'MAE', 'MAPE']:
                    fig_metrics.add_trace(go.Bar(
                        name=metric,
                        x=list(metrics_data.keys()),
                        y=[metrics_data[model][metric] for model in metrics_data.keys()]
                    ))
                fig_metrics.update_layout(
                    title='Error Metrics Comparison (Lower is Better)',
                    xaxis_title='Model',
                    yaxis_title='Error Value',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_metrics, width='stretch')
                
                st.subheader('Detailed Metrics Comparison')
                metrics_df = pd.DataFrame(metrics_data).T
                st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'), width='stretch')
            
