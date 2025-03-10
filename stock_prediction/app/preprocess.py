import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Preprocess input data
def preprocess_data(df, scaler_X):
    # Include all 10 features
    features = ['Open', 'High', 'Low', 'Close_Lag_1', 'MA_10', 'BB_Upper', 'MA_5', 'Cumulative_Return', 'Adj Close', 'Close']
    X = df[features].values
    X_scaled = scaler_X.transform(X)

    # Convert to LSTM format (sequences)
    sequence_length = 60
    X_sequences = []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i + sequence_length])

    return np.array(X_sequences)
