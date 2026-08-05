# SMS Spam Detection

Final project for the Machine Learning course.

## Setup

```bash
git clone git@github.com:kozyami/ML_final_project.git
cd ML_final_project
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"   # required before running
```

## How to run

```bash
jupyter notebook notebooks/   # run training notebooks
python app.py                 # launch Gradio UI on http://localhost:7860
```

### Docker

```bash
docker build -t sms-spam-detector .
docker run -p 7860:7860 sms-spam-detector
```

### Notebook order

1. `01_EDA_Preprocessing.ipynb` — data cleaning, TF-IDF vectorization, train/test split; outputs pickles to `data/`
2. `02a_NaiveBayes.ipynb` — MultinomialNB with GridSearchCV
3. `02b_LogisticRegression.ipynb` — Logistic Regression with GridSearchCV
4. `03_Comparison.ipynb` — side-by-side metrics, ROC curves, conclusion

Notebooks 2a and 2b can run in parallel after step 1 completes.

### Gradio UI features

- Text input for SMS messages
- Dropdown to switch between Naive Bayes and Logistic Regression models
- Pre-loaded example messages (spam + ham)

## Preprocessing pipeline

Lowercase → strip non-alphabetic characters → tokenize → remove NLTK English stopwords → PorterStemmer → filter 1-letter tokens → TF-IDF (max 3000 features, unigrams + bigrams, sublinear TF).

## Dataset

**UCI SMS Spam Collection** — 5,574 SMS messages: 4,827 ham (86.6%), 747 spam (13.4%).

## Team

| Person | Algorithm |
|--------|-----------|
| kozyami | Naive Bayes (MultinomialNB) |
| DaraDavit | Logistic Regression |

## Results

| Metric | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Accuracy | 0.9794 | **0.9803** |
| Precision | **0.9773** | 0.9635 |
| Recall | 0.8658 | **0.8859** |
| F1-Score | 0.9181 | **0.9231** |
| ROC-AUC | 0.9841 | **0.9856** |

Winner: **Logistic Regression** (higher F1 and recall for spam class). Full details in `03_Comparison.ipynb`.

## File structure

```
├── app.py
├── Dockerfile
├── README.md
├── requirements.txt
├── .dockerignore
├── .gitignore
├── SMS Spam Detection Presentation.pdf
├── data/
│   ├── SMSSpamCollection
│   ├── tfidf_vectorizer.pkl
│   ├── X_train_tfidf.pkl
│   ├── X_test_tfidf.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
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
