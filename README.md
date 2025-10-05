Hand Written Digit Recognition

This repository contains a Jupyter notebook (`Digit_Recognition.ipynb`) and a converted Python script (`digit_recognition.py`) that trains simple SVM and KNN pipelines on sklearn's digits dataset and saves the best model.

How to run locally

1. Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Run the training script:

```bash
python digit_recognition.py
```

This will train both models, print accuracy and classification reports, optionally show confusion matrix plots, and save the best pipeline as a `.joblib` file.

Git / GitHub

I initialized a local git repository and created an initial commit for you. To push to GitHub, add a remote and push:

```bash
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

If you'd like, provide the remote repo URL and I can add it and push for you (or I can show the exact commands to run).