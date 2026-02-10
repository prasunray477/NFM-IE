import pandas as pd
import yfinance as yf


# Function to fetch stock data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Handle timezone in Date column
        if 'Date' in data.columns:
            if data['Date'].dt.tz is not None:
                data['Date'] = data['Date'].dt.tz_localize(None)
            # Remove time component, keep only date
            data['Date'] = data['Date'].dt.date
        
        # Flatten multi-level columns if they exist (e.g., 'Close AAPL' -> 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
        
        # Rename columns to remove ticker suffix if present
        data.columns = [col.split()[0] if ' ' in str(col) else col for col in data.columns]
        
        return data
    except Exception as e:
        return None