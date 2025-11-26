## Why
Provide a simple, reproducible way to train and serve the myna-bird classifier as a small web app using Streamlit. This makes the model usable outside Colab and provides a canonical training script for reproducible model exports.

## What Changes
- Add a training script `scripts/train.py` that loads image folders, trains a transfer-learning ResNet50V2 model, and saves the exported model and label metadata.
- Add a Streamlit app `app.py` to load the exported model and provide an image-upload inference UI.
- Add `requirements.txt` listing runtime dependencies.

## Impact
- Affected specs: inference / app (new capability)
- Affected code: new files `scripts/train.py`, `app.py`, and `requirements.txt`.
- No breaking changes to existing notebooks. This change adds runnable artifacts and a small workflow for model training and serving.
