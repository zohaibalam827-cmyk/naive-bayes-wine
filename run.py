"""Run Gaussian Naive Bayes on the Wine dataset and print the results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB

from src.data import load_data
from src.naive_bayes import GaussianNaiveBayes

X, y, features = load_data()
print(f"Dataset: {len(X)} rows, {len(features)} features, classes: {sorted(set(y))}")

# 70% train / 30% test, keeping the class proportions equal
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = GaussianNaiveBayes().fit(X_train, y_train)
print("\nLearned class priors P(class):")
for c, p in zip(model.classes_, model.priors_):
    print(f"  {c}: {p:.3f}")

predictions = model.predict(X_test)
print(f"\nTest accuracy (our Naive Bayes):          {model.score(X_test, y_test):.4f}")
reference = GaussianNB().fit(X_train, y_train)
print(f"Test accuracy (scikit-learn GaussianNB):  {reference.score(X_test, y_test):.4f}")

# 5-fold cross-validation on all the data, for a steadier estimate
accs = []
for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
    accs.append(GaussianNaiveBayes().fit(X[tr], y[tr]).score(X[va], y[va]))
print(f"5-fold cross-validation accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")

labels = list(model.classes_)
cm = confusion_matrix(y_test, predictions, labels=labels)
print("\nConfusion matrix (rows = true class, columns = predicted):")
print("labels:", labels)
print(cm)
print("\n" + classification_report(y_test, predictions, labels=labels))

# One example prediction with its probabilities
probs = model.predict_proba(X_test[:1])[0]
print("Example wine (first test row):")
print("  true class:", y_test[0])
for c, p in zip(model.classes_, probs):
    print(f"  P({c} | wine) = {p:.4f}")

# Save the confusion matrix as a picture
Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)), labels)
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Naive Bayes on Wine: confusion matrix")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.colorbar(im)
fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=150)
print("\nSaved plot to results/confusion_matrix.png")
