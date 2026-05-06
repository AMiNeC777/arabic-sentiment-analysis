# Arabic Sentiment Analysis

Modular starter project for Arabic sentiment analysis experiments using:
- classical ML models (Naive Bayes, SVM, Logistic Regression),
- deep learning models (LSTM, CNN),
- transformer-based embeddings/models (AraBERT),
- reproducible experiments and evaluation utilities.

## Project Structure

- `data/`: raw, processed, and external datasets.
- `notebooks/`: EDA, preprocessing, and experiment notebooks.
- `src/`: source code modules (preprocessing, features, models, evaluation, utils).
- `experiments/`: experiment outputs and logs.
- `models/`: trained model artifacts and checkpoints.
- `reports/`: figures and final report assets.

## Quick Start

1. Create a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Adjust settings in `config.yaml`.
4. Run:
   - `python src/main.py`

## Notes

- This scaffold provides clean templates so you can plug in your datasets and training logic quickly.
- Replace placeholder implementations with dataset-specific logic.
