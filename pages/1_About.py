import streamlit as st
from plotly import graph_objs as go

st.title('Workflow of the NFM-IE Stock Price Forecasting Application')

st.markdown("""
## Overview
The **Neutrosophic Forecasting Model based on Information Entropy (NFM-IE)** is an advanced time series forecasting model that uses neutrosophic logic to handle uncertainty and inconsistency in stock price predictions.

---

## Workflow Steps
""")

with st.expander("Step 1: Data Acquisition", expanded=True):
    st.markdown("""
    **Input:**
    - Stock ticker symbol (e.g., AAPL, TSLA)
    - Start date and end date
    
    **Process:**
    - Fetch historical stock data from Yahoo Finance
    - Extract Close prices
    - Handle multi-level columns and timezone issues
    
    **Output:**
    - Time series data V = [V₁, V₂, ..., Vₙ]
    """)

with st.expander("Step 2: Fluctuation Calculation", expanded=True):
    st.markdown("""
    **Process:**
    - Calculate first-order differences: Uₜ = Vₜ - Vₜ₋₁
    - This captures daily price changes
    
    **Output:**
    - Fluctuation series U = [U₁, U₂, ..., Uₙ₋₁]
    
    **Formula:**
    ```
    Uₜ = Vₜ - Vₜ₋₁
    ```
    """)

with st.expander("Step 3: Neutrosophication", expanded=True):
    st.markdown("""
    **Process:**
    Convert each fluctuation Uₜ into a neutrosophic set with three memberships:
    
    **1. Truth Membership T(Uₜ)** - Upward trend:
    - T = 0 if Uₜ ≤ -0.5×len
    - T = (Uₜ / (1.5×len)) + 1/3 if -0.5×len < Uₜ < len
    - T = 1 if Uₜ ≥ len
    
    **2. Indeterminacy Membership I(Uₜ)** - Inconsistency:
    - Based on information entropy of last m fluctuations
    - Measures historical pattern inconsistency
    - I = E(Uₜ) / max_entropy
    
    **3. Falsity Membership F(Uₜ)** - Downward trend:
    - F = 1 if Uₜ ≤ -len
    - F = (-Uₜ / (1.5×len)) + 1/3 if -len < Uₜ < 0.5×len
    - F = 0 if Uₜ ≥ 0.5×len
    
    **Output:**
    - Neutrosophic Fluctuation Time Series (NFTS): Xₜ = (T, I, F)
    """)

with st.expander("Step 4: Information Entropy Calculation", expanded=True):
    st.markdown("""
    **Process:**
    1. Fuzzify last m fluctuations into 5 linguistic labels:
       - l₁: Very Low [Uₘᵢₙ, -1.5×len)
       - l₂: Low [-1.5×len, -0.5×len)
       - l₃: Equal [-0.5×len, 0.5×len)
       - l₄: High [0.5×len, 1.5×len)
       - l₅: Very High [1.5×len, Uₘₐₓ)
    
    2. Calculate probability distribution: p(lₙ)
    
    3. Compute entropy: E(Uₜ) = -Σ p(lₙ) × log₂(p(lₙ))
    
    **Output:**
    - Normalized entropy value for indeterminacy membership
    """)

with st.expander(" Step 5: Neutrosophic Logical Relationships (NLRs)", expanded=True):
    st.markdown("""
    **Process:**
    - Build relationships: Xₜ → Xₜ₊₁
    - Calculate Jaccard similarity between current state and historical states
    
    **Jaccard Similarity Formula:**
    ```
    J(Xₜ, Xⱼ) = (Tₜ×Tⱼ + Iₜ×Iⱼ + Fₜ×Fⱼ) / 
                 (Tₜ² + Iₜ² + Fₜ² + Tⱼ² + Iⱼ² + Fⱼ² - numerator)
    ```
    
    **Output:**
    - Similarity scores between current and historical patterns
    """)

with st.expander("Step 6: Weighted Aggregation", expanded=True):
    st.markdown("""
    **Process:**
    1. Filter similar patterns (similarity ≥ threshold, default 0.89)
    2. If no patterns meet threshold, use top 5 most similar
    3. Apply Weighted Arithmetic Averaging (WAA):
    
    **Formula:**
    ```
    Xₜ₊₁ = Σ(Sₜ,ⱼ × Xⱼ) / Σ(Sₜ,ⱼ)
    ```
    
    **Output:**
    - Predicted neutrosophic state Xₜ₊₁ = (T', I', F')
    """)

with st.expander("Step 7: Deneutrosophication", expanded=True):
    st.markdown("""
    **Process:**
    Convert neutrosophic prediction back to real price value:
    
    **Formula:**
    ```
    U'ₜ₊₁ = (T'ₜ₊₁ - F'ₜ₊₁) × len
    V'ₜ₊₁ = Vₜ + U'ₜ₊₁
    ```
    
    **Output:**
    - Forecasted price V'ₜ₊₁
    """)

with st.expander("Step 8: Evaluation", expanded=True):
    st.markdown("""
    **Metrics:**
    - **RMSE** (Root Mean Square Error): √(Σ(actual - forecast)² / n)
    - **MAE** (Mean Absolute Error): Σ|actual - forecast| / n
    - **MAPE** (Mean Absolute Percentage Error): Σ|actual - forecast| / actual / n
    - **Theil's U**: Measures forecast accuracy relative to naive forecast
    
    **Output:**
    - Performance metrics comparing actual vs predicted prices
    """)

st.markdown("""
---

## Model Architecture Diagram
""")

# Create workflow diagram
fig = go.Figure()

steps = [
    "Data Acquisition",
    "Fluctuation Calculation",
    "Neutrosophication",
    "Information Entropy Calculation",
    "Establishing NLRs based on Jaccard Similarity",
    "Weighted Aggregation Calculation",
    "Deneutrosophication",
    "Price Forecast"
]

y_positions = list(range(len(steps), 0, -1))

for i, (step, y) in enumerate(zip(steps, y_positions)):
    color = 'lightblue' if i % 2 == 0 else 'lightgreen'
    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[y],
        mode='markers+text',
        marker=dict(size=40, color=color, line=dict(width=2, color='darkblue')),
        text=step,
        textposition='middle right',
        textfont=dict(size=12, color='black'),
        showlegend=False
    ))
    
    if i < len(steps) - 1:
        fig.add_trace(go.Scatter(
            x=[0.5, 0.5],
            y=[y, y-0.8],
            mode='lines',
            line=dict(color='darkblue', width=3),
            showlegend=False
        ))
        fig.add_annotation(
            x=0.5,
            y=y-0.4,
            text='↓',
            showarrow=False,
            font=dict(size=20, color='darkblue')
        )

fig.update_layout(
    title='NFM-IE Model Workflow',
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    height=700,
    plot_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---

## Key Advantages

**Handles Uncertainty**: Neutrosophic logic captures truth, indeterminacy, and falsity simultaneously

**Information Entropy**: Measures historical pattern inconsistency for better predictions

**Pattern Matching**: Uses Jaccard similarity to find similar historical patterns

**Adaptive**: Weighted aggregation adapts to data characteristics

**No Training Required**: Rule-based approach, no gradient descent or backpropagation

---

## Parameters

- **m** (default=9): Order of information entropy (lookback window)
- **threshold** (default=0.89): Similarity threshold for pattern matching
- **len**: Average absolute fluctuation (calculated from data)

    - ( Currently Hardcoded in this model, but can be made user-configurable in future versions )

## Future Enhancements

    - Integration with financial news sentiment analysis for enhanced predictions
    - Hybrid models combining NFM-IE with Bi-LSTM or ARIMA for improved accuracy
    - User interface for parameter tuning and real-time forecasting    
            
""")
