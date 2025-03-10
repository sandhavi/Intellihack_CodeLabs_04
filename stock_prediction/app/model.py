from tensorflow.keras.models import load_model
import pickle

# Load the trained model
def load_trained_model():
    model = load_model('models/improved_lstm_model.keras')
    return model

# Load the scalers
def load_scalers():
    with open('scalers/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scalers/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    return scaler_X, scaler_y
