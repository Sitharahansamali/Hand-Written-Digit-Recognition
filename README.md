Hand Written Digit Recognition
Thank you for visiting this project. Below are clear instructions on how to run the code, what to expect, and how to use the saved model.

How to Run

1. Clone this repository:

```bash
git clone https://github.com/<your-username>/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell / Command Prompt)
.venv\\Scripts\\activate
pip install -r requirements.txt
```

3. Run the script:

```bash
python digit_recognition.py
```

What the script does

- Load and visualize the digits dataset (sklearn.datasets.load_digits)
- Split the dataset into training and test sets
- Train SVM and KNN classifiers
- Print classification reports and accuracy
- Show confusion matrices (if a display is available)
- Display some random predictions
- Save the best-performing model as a `.joblib` file

Results

Both models typically perform at ~98% accuracy on the test set. The script compares SVM and KNN and saves the best model as `digits_best_<model>_pipeline.joblib`.

Example output:

```
SVM accuracy: 0.981
KNN accuracy: 0.978

Saved best model as: digits_best_svm_pipeline.joblib
```

Visualizations

- Confusion Matrix for SVM
- Confusion Matrix for KNN
- Random Test Predictions with True Label, SVM Prediction, and KNN Prediction

Using the saved model

You can reload the saved pipeline and use it for predictions:

```python
import joblib
import numpy as np

# Load trained pipeline
model = joblib.load("digits_best_svm_pipeline.joblib")

# Example: predict one digit from the test set
# (If you want to reproduce this exactly you can load X from sklearn.datasets)
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
sample = X[0].reshape(1, -1)  # reshape to (1,64)
print("Predicted:", model.predict(sample))
```

Notes

- Model files (`*.joblib`) are ignored by `.gitignore` to avoid committing large binary files.
- If running on a headless server, the script will skip plotting the figures and still save the model.

Author

Sithara Hansamali
