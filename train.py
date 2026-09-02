import os
import glob
import csv
import argparse
import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".aif"}

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y


def create_release_dataset(directory):
    """Read mix labels from meta.csv and extract features."""
    metadata_path = os.path.join(directory, "meta.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    label_map = {"bona-fide": 0, "spoof": 1}
    X, y = [], []
    missing_files = 0
    skipped_labels = 0

    with open(metadata_path, "r", newline="", encoding="utf-8") as metadata_file:
        for row in csv.DictReader(metadata_file):
            filename = row.get("file", "")
            label_name = row.get("label", "").strip().lower()
            if label_name not in label_map or not filename:
                skipped_labels += 1
                continue

            audio_path = os.path.join(directory, filename)
            if not os.path.exists(audio_path):
                missing_files += 1
                continue

            mfcc_features = extract_mfcc_features(audio_path)
            if mfcc_features is not None:
                X.append(mfcc_features)
                y.append(label_map[label_name])

    print("Release dataset:", len(X), "audio files loaded")
    print("Release dataset class counts:", {"bona-fide": y.count(0), "spoof": y.count(1)})
    if missing_files:
        print("Release dataset files missing:", missing_files)
    if skipped_labels:
        print("Release dataset rows skipped:", skipped_labels)
    return X, y

def train_model(X, y, sources, validation_size=0.5):
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("Atleast 2 set is required to train")

    print("Size of X:", X.shape)
    print("Size of y:", y.shape)

    train_indexes = []
    validation_indexes = []
    for source in sorted(set(sources)):
        source_indexes = np.flatnonzero(np.asarray(sources) == source)
        source_train, source_validation = train_test_split(
            source_indexes,
            test_size=validation_size,
            random_state=42,
            stratify=np.asarray(y)[source_indexes],
        )
        train_indexes.extend(source_train)
        validation_indexes.extend(source_validation)

    train_indexes = np.asarray(train_indexes)
    validation_indexes = np.asarray(validation_indexes)
    X_train, X_test = X[train_indexes], X[validation_indexes]
    y_train, y_test = y[train_indexes], y[validation_indexes]
    sources_train, sources_test = sources[train_indexes], sources[validation_indexes]

    print("Size of X_train:", X_train.shape)
    print("Size of X_test:", X_test.shape)
    print("Size of y_train:", y_train.shape)
    print("Size of y_test:", y_test.shape)
    print("Training split:", 1 - validation_size)
    print("Validation split:", validation_size)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    source_counts = {source: np.sum(sources_train == source) for source in set(sources_train)}
    source_weight = {
        source: len(sources_train) / (len(source_counts) * count)
        for source, count in source_counts.items()
    }
    sample_weights = np.asarray([source_weight[source] for source in sources_train])
    svm_classifier = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
    svm_classifier.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    y_pred = svm_classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_mtx)
    print("Validation accuracy by source:")
    source_metrics = {}
    for source in sorted(set(sources_test)):
        source_mask = np.asarray(sources_test) == source
        source_accuracy = accuracy_score(y_test[source_mask], y_pred[source_mask])
        source_matrix = confusion_matrix(y_test[source_mask], y_pred[source_mask], labels=[0, 1])
        source_metrics[source] = {
            "accuracy": float(source_accuracy),
            "files": int(source_mask.sum()),
            "confusion_matrix": source_matrix.tolist(),
        }
        print(f"  {source}: {source_accuracy:.4f} ({source_mask.sum()} files)")

    metrics = {
        "validation_accuracy": float(accuracy),
        "confusion_matrix": confusion_mtx.tolist(),
        "validation_size": validation_size,
        "training_files": int(len(y_train)),
        "validation_files": int(len(y_test)),
        "validation_by_source": source_metrics,
    }

    model_filename = os.path.join(BASE_DIR, "model.pkl")
    scaler_filename = os.path.join(BASE_DIR, "scaler.pkl")
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)
    with open(os.path.join(BASE_DIR, "training_metrics.json"), "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

def main(include_release=True):
    real_dir = os.path.join(BASE_DIR, "dataset", "real")
    fake_dir = os.path.join(BASE_DIR, "dataset", "fake")

    X_real, y_real = create_dataset(real_dir, label=0)
    X_fake, y_fake = create_dataset(fake_dir, label=1)

    datasets = [(X_real, y_real, ["original"] * len(y_real)), (X_fake, y_fake, ["original"] * len(y_fake))]
    if include_release:
        release_dir = os.path.join(BASE_DIR, "dataset", "mix")
        release_features, release_labels = create_release_dataset(release_dir)
        datasets.append((release_features, release_labels, ["mix"] * len(release_labels)))

    if any(len(features) < 2 for features, _, _ in datasets):
        print("Each class should have at least two samples for stratified splitting.")
        print("Combining both classes into one for training.")
    X = np.vstack([features for features, _, _ in datasets])
    y = np.hstack([labels for _, labels, _ in datasets])
    sources = np.hstack([dataset_sources for _, _, dataset_sources in datasets])

    train_model(X, y, sources, validation_size=0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AI voice detector")
    parser.add_argument(
        "--without-release",
        action="store_true",
        help="Train only with dataset/real and dataset/fake",
    )
    args = parser.parse_args()
    main(include_release=not args.without_release)