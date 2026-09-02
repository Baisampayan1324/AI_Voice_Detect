import os
from pathlib import Path
import numpy as np
from train import extract_mfcc_features
import joblib

def load_model(base_dir=None):
    """Load the trained classifier and scaler from the project directory."""
    model_dir = os.fspath(base_dir) if base_dir else os.getcwd()
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("model.pkl and scaler.pkl are missing. Run python train.py first.")
    return joblib.load(model_path), joblib.load(scaler_path)


def analyze_audio(input_audio_path, model=None, scaler=None):
    """
    Analyze audio file and return prediction with confidence.
    
    Args:
        input_audio_path: Path to a supported audio file to analyze
        
    Returns:
        Tuple of (prediction_string, prediction_label, confidence)
        prediction_string: "real" or "fake"
        confidence: Float between 0 and 1
    """
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError("The specified file does not exist.")
    elif Path(input_audio_path).suffix.lower() not in {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff', '.aif', '.webm'}:
        raise ValueError("Unsupported audio format.")
    
    if model is None or scaler is None:
        model, scaler = load_model()

    mfcc_features = extract_mfcc_features(input_audio_path)

    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = model.predict(mfcc_features_scaled)[0]
        
        prediction_label = "real" if prediction == 0 else "fake"
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(mfcc_features_scaled)[0]))
        else:
            decision_score = float(model.decision_function(mfcc_features_scaled)[0])
            confidence = abs(decision_score) / (abs(decision_score) + 1)
        return prediction_label, confidence
    else:
        raise ValueError("Unable to process the input audio.")

if __name__ == "__main__":
    user_input_file = input("Enter the path of the file to analyze: ")
    try:
        prediction_label, confidence = analyze_audio(user_input_file)
        print(f"The input audio is classified as {prediction_label}.")
        print(f"Confidence: {confidence:.2%}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")