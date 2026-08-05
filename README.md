# SMS Spam Detection

Final project for the Machine Learning course.

## Dataset

**UCI SMS Spam Collection** — 5,574 SMS messages labeled as ham (legitimate) or spam.  
Download: `data/SMSSpamCollection`

## Team

| Person | Algorithm |
|--------|-----------|
| kozyami | Naive Bayes (MultinomialNB) |
| DaraDavit | Logistic Regression |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/           # run training notebooks
python app.py                         # launch Gradio UI (port 7860)
```

### Docker

```bash
docker build -t sms-spam-detector .
docker run -p 7860:7860 sms-spam-detector
```

Run notebooks in order:

1. `01_EDA_Preprocessing.ipynb` — exploratory analysis, cleaning, TF-IDF, train/test split
2. `02a_NaiveBayes.ipynb` — kozyami: Naive Bayes with GridSearch
3. `02b_LogisticRegression.ipynb` — DaraDavit: Logistic Regression with GridSearch
4. `03_Comparison.ipynb` — side-by-side comparison, ROC curves, conclusion

## Results

| Metric | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Accuracy | 0.9758 | 0.9785 |
| F1-Score | 0.9032 | **0.9167** |
| ROC-AUC | 0.9821 | **0.9838** |

Winner: **Logistic Regression** (better recall for spam class). Full details in `03_Comparison.ipynb`.

## File structure

```
├── app.py
├── Dockerfile
├── README.md
├── requirements.txt
├── .dockerignore
├── .gitignore
├── data/
│   ├── SMSSpamCollection
│   ├── tfidf_vectorizer.pkl
│   ├── model_nb.pkl
│   ├── model_lr.pkl
│   ├── metrics_nb.pkl
│   └── metrics_lr.pkl
└── notebooks/
    ├── 01_EDA_Preprocessing.ipynb
    ├── 02a_NaiveBayes.ipynb
    ├── 02b_LogisticRegression.ipynb
    └── 03_Comparison.ipynb
```
