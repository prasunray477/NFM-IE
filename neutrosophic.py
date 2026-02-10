import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard
from sklearn.metrics import mean_squared_error, mean_absolute_error
from load_dataset import load_data
import warnings
warnings.filterwarnings('ignore')


# DATA ACQUISITION & PREPARATION
def fetch_stock_data(ticker, start, end):    
    data = load_data(ticker, start, end)
    
    if data is None or data.empty:
        return np.array([])
    
    return data['Close'].values

def create_fluctuation_series(V):
    return np.diff(V)


#NEUTROSOPHICATION - CONVERT TO NEUTROSOPHIC SETS
def calculate_membership_params(U):
    return np.mean(np.abs(U))


def truth_membership(U_t, len_val):
    """
    Truth-membership T(U_t): upward trend
    T = 0 if U_t <= -0.5*len
    T = (U_t / (3/2*len)) + 1/3 if -0.5*len <= U_t <= len
    T = 1 if U_t >= len
    """
    if U_t <= -0.5 * len_val:
        return 0.0
    elif U_t >= len_val:
        return 1.0
    else:
        return (U_t / (1.5 * len_val)) + (1.0 / 3.0)


def falsity_membership(U_t, len_val):
    """
    Falsity-membership F(U_t): downward trend
    F = 1 if U_t <= -len
    F = (-U_t / (3/2*len)) + 1/3 if -len <= U_t <= 0.5*len
    F = 0 if U_t >= 0.5*len
    """
    if U_t <= -len_val:
        return 1.0
    elif U_t >= 0.5 * len_val:
        return 0.0
    else:
        return (-U_t / (1.5 * len_val)) + (1.0 / 3.0)



#INFORMATION ENTROPY OF m-th ORDER FLUCTUATION
def fuzzify_fluctuation(U, len_val):
    """
    Fuzzify fluctuation into linguistic set L = {l1, l2, l3, l4, l5}
    l1: [U_min, -1.5*len)     - very low
    l2: [-1.5*len, -0.5*len)  - low
    l3: [-0.5*len, 0.5*len)   - equal
    l4: [0.5*len, 1.5*len)    - high
    l5: [1.5*len, U_max)      - very high
    """
    if U < -1.5 * len_val:
        return 0
    elif U < -0.5 * len_val:
        return 1
    elif U < 0.5 * len_val:
        return 2
    elif U < 1.5 * len_val:
        return 3
    else:
        return 4


def calculate_information_entropy(U_history, len_val, m=9):
    """
    Calculate information entropy of m-th order fluctuation
    E(U_t) = -Σ p(L_n) * log2(p(L_n))
    """
    if len(U_history) < m:
        return 0.0

    # Get last m fluctuations
    recent = U_history[-m:]

    # Fuzzify to linguistic labels
    labels = [fuzzify_fluctuation(u, len_val) for u in recent]

    # Calculate probabilities
    counts = np.bincount(labels, minlength=5)
    probs = counts / m

    # Calculate entropy (avoid log(0))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def normalize_entropy(entropy, max_entropy=3.7):
    """Normalize entropy to [0,1] range"""
    return entropy / max_entropy

def indeterminacy_membership(U_history, len_val, m=9):
    """
    Indeterminacy-membership I(U_t): inconsistency of historical fluctuations
    Based on normalized information entropy
    """
    entropy = calculate_information_entropy(U_history, len_val, m)
    return normalize_entropy(entropy)


# BUILD NEUTROSOPHIC FLUCTUATION TIME SERIES (NFTS)
def build_nfts(U, m=9):
    """
    Build Neutrosophic Fluctuation Time Series
    X_t = (T(U_t), I(U_t), F(U_t))
    """
    len_val = calculate_membership_params(U)
    NFTS = []

    for t in range(m, len(U)):
        T = truth_membership(U[t], len_val)
        I = indeterminacy_membership(U[max(0, t-m):t], len_val, m)
        F = falsity_membership(U[t], len_val)
        NFTS.append((T, I, F))

    return np.array(NFTS), len_val

#NEUTROSOPHIC LOGICAL RELATIONSHIPS (NLRs)
def jaccard_similarity(X_t, X_j):
    """
    Jaccard similarity between two neutrosophic sets
    J(X_t, X_j) = (T_t*T_j + I_t*I_j + F_t*F_j) /
                  (T_t^2 + I_t^2 + F_t^2 + T_j^2 + I_j^2 + F_j^2 -
                   T_t*T_j - I_t*I_j - F_t*F_j)
    """
    T_t, I_t, F_t = X_t
    T_j, I_j, F_j = X_j

    numerator = T_t*T_j + I_t*I_j + F_t*F_j
    denominator = (T_t**2 + I_t**2 + F_t**2 +
                   T_j**2 + I_j**2 + F_j**2 - numerator)

    if denominator == 0:
        return 0.0

    return numerator / denominator


#AGGREGATION OPERATOR FOR FORECASTING"""
def weighted_aggregation(similarities, D_values, threshold=0.89):
    """
    Weighted Arithmetic Averaging (WAA) aggregation operator
    D_j = Σ(S_Xt,j × D_t) / Σ(S_Xt,j)
    """
    # Filter by threshold
    valid_idx = similarities >= threshold

    if not np.any(valid_idx):
        # If no similar patterns, use top 5
        top_k = min(5, len(similarities))
        valid_idx = np.argsort(similarities)[-top_k:]
        valid_sims = similarities[valid_idx]
        valid_D = D_values[valid_idx]
    else:
        valid_sims = similarities[valid_idx]
        valid_D = D_values[valid_idx]

    if len(valid_sims) == 0:
        return np.mean(D_values, axis=0)

    # Weighted average
    sum_sims = np.sum(valid_sims)
    T_pred = np.sum(valid_sims * valid_D[:, 0]) / sum_sims
    I_pred = np.sum(valid_sims * valid_D[:, 1]) / sum_sims
    F_pred = np.sum(valid_sims * valid_D[:, 2]) / sum_sims

    return np.array([T_pred, I_pred, F_pred])

#DENEUTROSOPHICATION & FORECASTING
def deneutrosophicate(X_pred, len_val, V_current):
    """
    Convert neutrosophic prediction back to real value
    V'_{t+1} = (T_{t+1} - F_{t+1}) × len + V_t
    """
    T_pred, I_pred, F_pred = X_pred
    U_pred = (T_pred - F_pred) * len_val
    V_pred = V_current + U_pred
    return V_pred



# EVALUATION METRICS
def calculate_metrics(actual, forecast):
    """Calculate RMSE, MSE, MAE, MAPE, Theil's U"""
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual))

    # Theil's U statistic
    numerator = np.sqrt(np.mean((forecast - actual)**2))
    denominator = np.sqrt(np.mean(forecast**2)) + np.sqrt(np.mean(actual**2))
    theil_u = numerator / denominator if denominator != 0 else 0

    return {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape,
        'Theil_U': theil_u
    }

#neutrosophic points mapping
def map_neutrosophic_points(nfts):
    """Map neutrosophic points for visualization"""
    T_vals = nfts[:, 0]
    I_vals = nfts[:, 1]
    F_vals = nfts[:, 2]
    return T_vals, I_vals, F_vals



class NFM_IE:
    """Neutrosophic Forecasting Model based on Information Entropy"""

    def __init__(self, m=9, threshold=0.89):
        self.m = m  # Order of information entropy
        self.threshold = threshold  # Similarity threshold
        self.len_val = None
        self.NFTS_train = None
        self.V_train = None
        self.U_train = None

    def fit(self, V_train):
        """Train the model on historical data"""
        self.V_train = V_train
        self.U_train = create_fluctuation_series(V_train)
        self.NFTS_train, self.len_val = build_nfts(self.U_train, self.m)

    def predict(self, V_history, steps=1):
        """Forecast next 'steps' values"""
        predictions = []
        V_current = V_history.copy()

        for _ in range(steps):
            # Get current fluctuation series
            U_current = create_fluctuation_series(V_current)

            # Build current neutrosophic state
            if len(U_current) < self.m:
                # Not enough history, use simple average
                predictions.append(V_current[-1])
                V_current = np.append(V_current, V_current[-1])
                continue

            T_curr = truth_membership(U_current[-1], self.len_val)
            I_curr = indeterminacy_membership(U_current[-self.m:], self.len_val, self.m)
            F_curr = falsity_membership(U_current[-1], self.len_val)
            X_current = np.array([T_curr, I_curr, F_curr])

            # Calculate similarities with training data
            similarities = np.array([
                jaccard_similarity(X_current, X_train)
                for X_train in self.NFTS_train[:-1]
            ])

            # Get corresponding next states (RHS of NLRs)
            D_values = self.NFTS_train[1:]

            # Aggregate to get prediction
            X_pred = weighted_aggregation(similarities, D_values, self.threshold)

            # Deneutrosophicate
            V_pred = deneutrosophicate(X_pred, self.len_val, V_current[-1])

            predictions.append(V_pred)
            V_current = np.append(V_current, V_pred)

        return np.array(predictions)
    