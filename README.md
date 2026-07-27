# SMS Spam Detection

Final project for the Machine Learning course.

## Dataset

**UCI SMS Spam Collection** — 5,574 SMS messages labeled as ham (legitimate) or spam.  
Download: `data/SMSSpamCollection`

## Team

| Person | Algorithm |
|--------|-----------|
| A | Naive Bayes (MultinomialNB) |
| B | Logistic Regression |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run notebooks in order:

1. `01_EDA_Preprocessing.ipynb` — exploratory analysis, cleaning, TF-IDF, train/test split
2. `02a_NaiveBayes.ipynb` — Person A: Naive Bayes with GridSearch
3. `02b_LogisticRegression.ipynb` — Person B: Logistic Regression with GridSearch
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
├── AGENTS.md
├── Final-Project-Instructions.txt
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── SMSSpamCollection
└── notebooks/
    ├── 01_EDA_Preprocessing.ipynb
    ├── 02a_NaiveBayes.ipynb
    ├── 02b_LogisticRegression.ipynb
    └── 03_Comparison.ipynb
```
