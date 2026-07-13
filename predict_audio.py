from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd


def extract_audio_features(
    file_path: Path,
    target_sample_rate: int = 22050,
    number_of_mfccs: int = 13,
) -> dict[str, float]:
    """Extract the same features used during model training."""

    signal, sample_rate = librosa.load(
        file_path,
        sr=target_sample_rate,
        mono=True,
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=number_of_mfccs,
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate,
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sample_rate,
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sample_rate,
        roll_percent=0.85,
    )
    rms_energy = librosa.feature.rms(y=signal)

    features: dict[str, float] = {}

    for index in range(number_of_mfccs):
        features[f"mfcc_{index + 1}_mean"] = float(
            np.mean(mfcc[index])
        )
        features[f"mfcc_{index + 1}_std"] = float(
            np.std(mfcc[index])
        )

    features["zero_crossing_rate_mean"] = float(
        np.mean(zero_crossing_rate)
    )
    features["spectral_centroid_mean"] = float(
        np.mean(spectral_centroid)
    )
    features["spectral_bandwidth_mean"] = float(
        np.mean(spectral_bandwidth)
    )
    features["spectral_rolloff_mean"] = float(
        np.mean(spectral_rolloff)
    )
    features["rms_energy_mean"] = float(
        np.mean(rms_energy)
    )

    return features


def predict_audio(
    audio_path: Path,
    model_path: Path,
) -> None:
    """Load the trained model and predict one audio file."""

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    extracted_features = extract_audio_features(audio_path)

    feature_vector = pd.DataFrame(
        [
            {
                column: extracted_features[column]
                for column in feature_columns
            }
        ],
        columns=feature_columns,
    )

    predicted_class = model.predict(feature_vector)[0]

    print()
    print(f"Audio file: {audio_path.name}")
    print(f"Predicted class: {predicted_class}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)[0]
        class_names = model.classes_

        ranked_predictions = sorted(
            zip(class_names, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        print()
        print("Top 10 prediction probabilities:")

        for class_name, probability in ranked_predictions[:10]:
            print(
                f"  {class_name:25s}: "
                f"{probability * 100:6.2f}%"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the ESC-50 environmental sound class "
            "of a WAV file."
        )
    )

    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the WAV file",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/random_forest_esc50_classifier.joblib"
        ),
        help="Path to the saved model",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    predict_audio(
        audio_path=args.audio_file,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
