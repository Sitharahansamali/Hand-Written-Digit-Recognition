import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_data():
    """Load the sklearn digits dataset."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y, digits


def build_pipelines():
    """Create SVM and KNN training pipelines."""
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=42))
    ])

    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    return svm_pipeline, knn_pipeline


def train_and_evaluate(X, y):
    """Train both pipelines and evaluate on a held-out test set.

    Returns a dict with results and the trained pipeline objects.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    svm_pipeline, knn_pipeline = build_pipelines()

    svm_pipeline.fit(x_train, y_train)
    knn_pipeline.fit(x_train, y_train)

    y_pred_svm = svm_pipeline.predict(x_test)
    y_pred_knn = knn_pipeline.predict(x_test)

    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    results = {
        'svm': {
            'pipeline': svm_pipeline,
            'accuracy': accuracy_svm,
            'classification_report': classification_report(y_test, y_pred_svm, digits=3),
            'confusion_matrix': confusion_matrix(y_test, y_pred_svm)
        },
        'knn': {
            'pipeline': knn_pipeline,
            'accuracy': accuracy_knn,
            'classification_report': classification_report(y_test, y_pred_knn, digits=3),
            'confusion_matrix': confusion_matrix(y_test, y_pred_knn)
        }
    }

    return results, x_test, y_test


def save_best_model(results, out_dir='.', prefix='digits_best'):
    """Pick the best model by accuracy and save it as a joblib file.

    Returns the filename written.
    """
    if results['svm']['accuracy'] >= results['knn']['accuracy']:
        best = 'svm'
    else:
        best = 'knn'

    filename = f"{prefix}_{best}_pipeline.joblib"
    filepath = f"{out_dir.rstrip('/')}/{filename}"
    joblib.dump(results[best]['pipeline'], filepath)
    return filepath


def plot_confusion(cm, title=None, cmap='Greens'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if title:
        plt.title(title)
    plt.show()


def main():
    X, y, digits = load_data()
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('Unique labels:', np.unique(y))

    results, x_test, y_test = train_and_evaluate(X, y)

    print(f"SVM accuracy: {results['svm']['accuracy']:.4f}")
    print(f"KNN accuracy: {results['knn']['accuracy']:.4f}")

    print('\nSVM classification report:\n', results['svm']['classification_report'])
    print('\nKNN classification report:\n', results['knn']['classification_report'])

    # Plot confusion matrices (optional - comment out if running headless)
    try:
        plot_confusion(results['svm']['confusion_matrix'], title='Confusion Matrix - SVM', cmap='Greens')
        plot_confusion(results['knn']['confusion_matrix'], title='Confusion Matrix - KNN', cmap='Blues')
    except Exception:
        # If running in an environment without a display, skip plotting
        pass

    saved = save_best_model(results)
    print('Saved best model as:', saved)


if __name__ == '__main__':
    main()
