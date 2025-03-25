import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ------------------------------
# Streamlit App Setup
# ------------------------------
st.set_page_config(layout="wide")
st.title("📊 Multi-Factor Quant Model (Smart Beta)")
st.markdown("""
This app ranks stocks using **value, quality, momentum, and low-volatility factors** combined with technical triggers.
""")

# ------------------------------
# 1. Load Stock Universe from Excel
# ------------------------------
try:
    stock_data = pd.ExcelFile("stocklist1.xlsx")  # Replace with your file path
    sheet_names = stock_data.sheet_names
    
    selected_sheet = st.sidebar.selectbox(
        "**Select Stock Index**",
        options=sheet_names,
        index=0,
        help="Choose NIFTY50, NIFTY100, etc."
    )
    
    df_stocks = pd.read_excel(stock_data, sheet_name=selected_sheet)
    tickers = [ticker + ".NS" for ticker in df_stocks["Symbol"].tolist()]  # Add .NS suffix
    st.sidebar.success(f"Loaded {len(tickers)} stocks from {selected_sheet}")

except Exception as e:
    st.error(f"Error loading Excel file: {e}")
    st.stop()

# ------------------------------
# 2. User Inputs
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    # Risk Tolerance → Factor Weights
    st.subheader("⚖️ Risk Profile")
    risk_tolerance = st.radio(
        "Risk Tolerance",
        options=["Low", "Medium", "High"],
        index=1,
        help="Adjusts factor weights (more quality/low-vol for low risk)"
    )
    
    # Time Horizon → Factor Priority
    time_horizon = st.radio(
        "Time Horizon",
        options=["Short-Term (1-3M)", "Medium-Term (3-6M)", "Long-Term (6M+)"],
        index=1,
        help="Short-term favors momentum, long-term favors value"
    )

with col2:
    # Technical Filters
    st.subheader("📈 Technical Filters")
    min_rsi = st.slider("Minimum RSI", 30, 60, 40)
    min_volume = st.number_input("Minimum Avg Volume (Millions)", value=1.0)
    momentum_lookback = st.selectbox(
        "Momentum Lookback Period", 
        options=["30D", "60D", "90D", "180D"],
        index=2
    )

# ------------------------------
# 3. Factor Weighting Logic
# ------------------------------
# Define factor weights based on user inputs
if risk_tolerance == "Low":
    weights = {
        'quality': 0.4, 
        'low_vol': 0.3,
        'value': 0.2,
        'momentum': 0.1
    }
elif risk_tolerance == "Medium":
    weights = {
        'quality': 0.3,
        'value': 0.3,
        'momentum': 0.2,
        'low_vol': 0.2
    }
else:  # High risk
    weights = {
        'momentum': 0.4,
        'value': 0.3,
        'quality': 0.2,
        'low_vol': 0.1
    }

# Adjust for time horizon
if "Short" in time_horizon:
    weights['momentum'] += 0.1
    weights['value'] -= 0.1
elif "Long" in time_horizon:
    weights['value'] += 0.15
    weights['momentum'] -= 0.1

# Normalize weights to sum to 1
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

# ------------------------------
# 4. Fetch Data & Calculate Factors
# ------------------------------
@st.cache_data
def get_factor_scores(tickers):
    data = []
    for ticker in tickers[:50]:  # Limit to 50 for demo
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get history for technicals
            hist = stock.history(period="6mo")
            if hist.empty:
                continue
                
            # Calculate technical factors
            returns_6m = (hist['Close'][-1] / hist['Close'][0] - 1) * 100 if len(hist) > 0 else np.nan
            
            # Simplified RSI calculation
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not loss.isna().all() else 50
            
            volume_avg = hist['Volume'].mean() / 1e6  # In millions
            
            # Get fundamentals
            pe = info.get('trailingPE', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            de = info.get('debtToEquity', np.nan)
            market_cap = info.get('marketCap', np.nan) / 1e9  # In $B
            
            data.append({
                'Ticker': ticker.replace(".NS", ""),
                'P/E': pe,
                'ROE': roe * 100 if roe else np.nan,  # Convert to percentage
                'Debt/Equity': de,
                '6M Momentum': returns_6m,
                'RSI': rsi,
                'Volume (M)': volume_avg,
                'Market Cap': market_cap
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(data)

if st.button("Run Analysis"):
    with st.spinner(f"Analyzing {selected_sheet} stocks..."):
        df = get_factor_scores(tickers)
        
        if df.empty:
            st.error("No data could be fetched for any stocks. Please try again later.")
            st.stop()
        
        # Ensure required columns exist
        required_cols = ['P/E', 'ROE', '6M Momentum', 'RSI', 'Volume (M)']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Filter stocks
        df = df[
            (df['RSI'].notna()) & 
            (df['Volume (M)'].notna()) & 
            (df['P/E'].notna())
        ].copy()
        
        if df.empty:
            st.warning("No stocks passed all filters. Try relaxing criteria.")
            st.stop()
        
        # Normalize factors (0-1 scale)
        factors = {
            'value': 1/df['P/E'],  # Inverse P/E (higher = better)
            'quality': df['ROE'],
            'momentum': df['6M Momentum'],
            'low_vol': -df['6M Momentum'].abs()  # Lower absolute momentum = less volatile
        }
        
        # Calculate composite score
        for factor in factors:
            if factors[factor].notna().any():
                min_val = factors[factor].min()
                max_val = factors[factor].max()
                if max_val > min_val:  # Avoid division by zero
                    df[factor] = (factors[factor] - min_val) / (max_val - min_val)
                else:
                    df[factor] = 0.5  # Neutral score if all values are equal
            else:
                df[factor] = 0.5  # Neutral score if no data
        
        df['Score'] = sum(df[factor] * weights[factor] for factor in weights)
        df = df.sort_values('Score', ascending=False).head(20)
        
        # Add allocation % (based on score)
        if df['Score'].sum() > 0:
            df['Allocation (%)'] = (df['Score'] / df['Score'].sum() * 100).round(1)
        else:
            df['Allocation (%)'] = 100 / len(df)  # Equal allocation if all scores zero
        
        # Format display
        df_display = df[[
            'Ticker', 'Score', 'Allocation (%)',
            'P/E', 'ROE', '6M Momentum', 'RSI'
        ]].rename(columns={
            '6M Momentum': 'Momentum (%)',
            'ROE': 'ROE (%)'
        })
        
        # ------------------------------
        # 5. Display Results
        # ------------------------------
        st.success(f"Top {len(df)} Stocks for {risk_tolerance} Risk / {time_horizon}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.dataframe(
                df_display.style.format({
                    'Score': '{:.2f}',
                    'Allocation (%)': '{:.1f}%',
                    'P/E': '{:.1f}',
                    'ROE (%)': '{:.1f}%',
                    'Momentum (%)': '{:.1f}%',
                    'RSI': '{:.1f}'
                }).background_gradient(
                    subset=['Score', 'Allocation (%)'],
                    cmap='Blues'
                ),
                height=600
            )
        
        with col2:
            st.subheader("Factor Weights")
            for factor, weight in weights.items():
                st.metric(
                    label=factor.capitalize(),
                    value=f"{weight*100:.1f}%"
                )
            
            st.subheader("Key Metrics")
            st.metric("Avg P/E", f"{df['P/E'].mean():.1f}")
            st.metric("Avg Momentum", f"{df['6M Momentum'].mean():.1f}%")
            st.metric("Total Allocation", f"{df['Allocation (%)'].sum():.1f}%")

# ------------------------------
# 6. Footer
# ------------------------------
st.markdown("---")
st.markdown("""
**Methodology**:
- **Value**: Inverse P/E ratio
- **Quality**: Return on Equity (ROE)
- **Momentum**: 6-month price return
- **Low Volatility**: Negative absolute momentum
""")
