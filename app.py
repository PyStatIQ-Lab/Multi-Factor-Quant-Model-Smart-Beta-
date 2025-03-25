import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Load stock list
@st.cache_data
def load_stocklist():
    file_path = "stocklist.xlsx"
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names  # Get sheet names
    return {sheet: pd.read_excel(xls, sheet_name=sheet)['Symbol'].tolist() for sheet in sheets}

# Fetch fundamental & technical data from yfinance
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Fundamental Factors
        pe_ratio = info.get('trailingPE', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        debt_to_equity = info.get('debtToEquity', np.nan)
        earnings_growth = info.get('earningsGrowth', np.nan)
        
        # Technical Factors
        hist = stock.history(period="6mo")
        if not hist.empty:
            six_month_momentum = (hist['Close'][-1] / hist['Close'][0]) - 1  # % Change
            rsi = 100 - (100 / (1 + (hist['Close'].pct_change().dropna().mean() / hist['Close'].pct_change().dropna().std())))
            volume_surge = hist['Volume'][-1] / hist['Volume'].rolling(20).mean()[-1]
        else:
            six_month_momentum = np.nan
            rsi = np.nan
            volume_surge = np.nan
        
        return {
            "Symbol": symbol,
            "P/E Ratio": pe_ratio,
            "ROE": roe,
            "Debt/Equity": debt_to_equity,
            "Earnings Growth": earnings_growth,
            "6M Momentum": six_month_momentum,
            "RSI": rsi,
            "Volume Surge": volume_surge
        }
    except Exception as e:
        return None

# Score Calculation & Ranking
def calculate_scores(df, risk_tolerance, time_horizon):
    # Normalize & assign weights based on user inputs
    df = df.dropna().reset_index(drop=True)
    
    df["Fundamental Score"] = (df["P/E Ratio"].rank(ascending=False) +
                               df["ROE"].rank(ascending=True) +
                               df["Debt/Equity"].rank(ascending=False) +
                               df["Earnings Growth"].rank(ascending=True))
    
    df["Technical Score"] = (df["6M Momentum"].rank(ascending=True) +
                             df["RSI"].rank(ascending=True) +
                             df["Volume Surge"].rank(ascending=True))
    
    # Adjust weights based on Risk & Time Horizon
    fundamental_weight = 0.7 if time_horizon == "Long-Term" else 0.4
    technical_weight = 0.3 if time_horizon == "Long-Term" else 0.6
    
    if risk_tolerance == "Low":
        fundamental_weight += 0.1
        technical_weight -= 0.1
    elif risk_tolerance == "High":
        fundamental_weight -= 0.1
        technical_weight += 0.1

    df["Final Score"] = df["Fundamental Score"] * fundamental_weight + df["Technical Score"] * technical_weight
    df = df.sort_values(by="Final Score", ascending=False)
    
    return df

# Streamlit UI
st.title("📈 Multi-Factor Quant Model (Smart Beta)")

# Load stocklist
stocklist = load_stocklist()
sheet_selection = st.selectbox("Select Stock List", options=list(stocklist.keys()))

# User Inputs
risk_tolerance = st.radio("Select Risk Tolerance", ["Low", "Medium", "High"], index=1)
time_horizon = st.radio("Select Time Horizon", ["Short-Term", "Long-Term"], index=1)

# Fetch data for selected stocks
symbols = stocklist[sheet_selection]
st.write(f"Fetching data for {len(symbols)} stocks...")

stock_data = [get_stock_data(symbol) for symbol in symbols]
stock_df = pd.DataFrame([s for s in stock_data if s])

# Check if data exists
if not stock_df.empty:
    ranked_df = calculate_scores(stock_df, risk_tolerance, time_horizon)
    
    # Display top stocks
    st.subheader("🏆 Top Ranked Stocks")
    st.dataframe(ranked_df[["Symbol", "Final Score"]].head(10))
    
    # Portfolio Weights
    st.subheader("📊 Optimized Portfolio Weights")
    ranked_df["Weight"] = ranked_df["Final Score"] / ranked_df["Final Score"].sum()
    st.dataframe(ranked_df[["Symbol", "Weight"]].head(10))

else:
    st.warning("No stock data found. Try selecting another sheet or check stock symbols.")
