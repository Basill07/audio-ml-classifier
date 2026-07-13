from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    /  "random_forest_esc50_classifier.joblib"
)


@st.cache_resource
def load_model_bundle() -> dict:
    """Load the saved model bundle once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def extract_audio_features(
    file_path: Path,
    target_sample_rate: int = 22050,
    number_of_mfccs: int = 13,
) -> dict[str, float]:
    """Extract the same features used during training."""

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

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        signal
    )
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


def create_feature_dataframe(
    extracted_features: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create a one-row DataFrame with training feature names."""

    return pd.DataFrame(
        [
            {
                column: extracted_features[column]
                for column in feature_columns
            }
        ],
        columns=feature_columns,
    )


def main() -> None:
    st.set_page_config(
        page_title="Audio Sound Classifier",
        page_icon="🎧",
        layout="wide",
    )

    st.title("Environmental Sound Classifier")
    st.write(
       "Upload a WAV file and classify it into one of the "
"50 ESC-50 environmental sound categories."
    )

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav"],
    )

    if uploaded_file is None:
        st.info("Upload a WAV file to begin.")
        return

    st.audio(
        uploaded_file.getvalue(),
        format="audio/wav",
    )

    try:
        model_bundle = load_model_bundle()
        model = model_bundle["model"]
        feature_columns = model_bundle["feature_columns"]

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = Path(temporary_file.name)

        signal, sample_rate = librosa.load(
            temporary_path,
            sr=22050,
            mono=True,
        )

        duration = len(signal) / sample_rate

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Duration",
                f"{duration:.2f} seconds",
            )

        with col2:
            st.metric(
                "Sample rate",
                f"{sample_rate} Hz",
            )

        st.subheader("Waveform")

        figure, axis = plt.subplots(figsize=(12, 4))
        librosa.display.waveshow(
            signal,
            sr=sample_rate,
            ax=axis,
        )
        axis.set_xlabel("Time in seconds")
        axis.set_ylabel("Amplitude")
        axis.set_title(uploaded_file.name)
        figure.tight_layout()

        st.pyplot(figure)
        plt.close(figure)

        extracted_features = extract_audio_features(
            temporary_path
        )

        feature_dataframe = create_feature_dataframe(
            extracted_features,
            feature_columns,
        )

        predicted_class = model.predict(
            feature_dataframe
        )[0]

        probabilities = model.predict_proba(
            feature_dataframe
        )[0]

        probability_dataframe = pd.DataFrame(
            {
                "Sound class": model.classes_,
                "Probability": probabilities,
            }
        ).sort_values(
            "Probability",
            ascending=False,
        )

        highest_probability = float(
            probability_dataframe.iloc[0]["Probability"]
        )

        st.subheader("Prediction")
        st.success(
            f"Predicted sound: {predicted_class.replace('_', ' ').title()}"
        )

        st.metric(
            "Highest model vote",
            f"{highest_probability * 100:.2f}%",
        )

        st.subheader("Class probabilities")

        chart_data = probability_dataframe.set_index(
            "Sound class"
        )
        st.bar_chart(chart_data)

        display_table = probability_dataframe.copy()
        display_table["Probability"] = (
            display_table["Probability"] * 100
        ).map(lambda value: f"{value:.2f}%")

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )

        if highest_probability < 0.50:
            st.warning(
                "The model is uncertain about this recording. "
                "The sound may not belong to one of the five "
                "trained classes."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

    finally:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
