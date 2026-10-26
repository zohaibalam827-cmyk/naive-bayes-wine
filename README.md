# Naive Bayes on the Wine Dataset

A from-scratch implementation of the Gaussian Naive Bayes classifier in Python, applied to the Wine dataset (178 wines, 13 chemical measurements, 3 grape varieties).

## How Naive Bayes works

Naive Bayes is a Bayesian network in which the class is the only parent of every feature. It assumes the features are independent once the class is known, and uses Bayes' rule:

    P(class | x)  is proportional to  P(class) * product of P(x_i | class)

1. **Training:** for each class, store its prior P(class), and the mean and variance of every feature.
2. **Prediction:** for a new wine, score every class with its prior times the Gaussian probability of each measurement, then pick the class with the highest score.
3. Logarithms are added instead of multiplying probabilities, so very small numbers don't round to zero.

## Results

- Test accuracy: 1.00 (matches scikit-learn's GaussianNB)
- 5-fold cross-validation accuracy: 0.9719 (+/- 0.0252)

## What this project does

- Loads Wine and saves it as `data/wine.csv`
- Splits it 70% train / 30% test, keeping the class balance
- Reports test accuracy, a confusion matrix and a classification report
- Compares with scikit-learn's GaussianNB
- Runs 5-fold cross-validation
- Saves the confusion matrix as `results/confusion_matrix.png`

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run

    python run.py

## Tests

    python -m pytest -v

## Files

- `src/naive_bayes.py` : the Gaussian Naive Bayes classifier
- `src/data.py` : data loading
- `run.py` : runs the full experiment
- `tests/test_naive_bayes.py` : pytest tests
- `data/wine.csv` : the dataset
- `results/` : output plot
