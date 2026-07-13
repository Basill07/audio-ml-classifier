<<<<<<< HEAD
# Machine Learning-Based Audio Classification

A beginner-friendly environmental sound classification project using the ESC-50 dataset, audio feature extraction with Librosa, and a Random Forest classifier.

## Initial classes

- Dog
- Rain
- Crying baby
- Clock tick
- Helicopter

The selected subset contains 200 recordings in total, because ESC-50 provides 40 recordings for each class.

## Project plan

1. Explore audio recordings
2. Extract MFCC and spectral features
3. Train a Random Forest classifier
4. Evaluate using ESC-50 folds
5. Save the model and create a simple prediction script

## Repository structure

```text
audio-ml-classifier/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_audio_exploration.ipynb
├── outputs/
│   └── figures/
├── src/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup on Windows

Open PowerShell or the VS Code terminal inside the project folder.

### 1. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Download ESC-50

```powershell
git clone https://github.com/karolpiczak/ESC-50.git data/raw/ESC-50
```

After cloning, this file should exist:

```text
data/raw/ESC-50/meta/esc50.csv
```

### 4. Start JupyterLab

```powershell
jupyter lab
```

Open:

```text
notebooks/01_audio_exploration.ipynb
```

Run the cells from top to bottom.
=======
# audio-ml-classifier
A beginner- friendly machine learning project for classifying environmental sounds using audio feature extraction, MFCCs, Librosa, and a Random Forest classifier 
>>>>>>> 7fe55fc3bfb4a501f07bff330effc9a14713fc05
