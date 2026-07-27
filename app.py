import re
import joblib
import gradio as gr
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

MODEL_NB = joblib.load("data/model_nb.pkl")
MODEL_LR = joblib.load("data/model_lr.pkl")
TFIDF = joblib.load("data/tfidf_vectorizer.pkl")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def classify(message: str, model_choice: str) -> dict:
    if not message.strip():
        return {"HAM": 0.0, "SPAM": 0.0}

    model = MODEL_LR if model_choice == "Logistic Regression" else MODEL_NB
    cleaned = clean_text(message)
    vec = TFIDF.transform([cleaned])
    proba = model.predict_proba(vec)[0]

    return {"HAM": float(proba[0]), "SPAM": float(proba[1])}


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(
            label="SMS Message",
            placeholder="Type or paste an SMS message here...",
            lines=4,
        ),
        gr.Dropdown(
            choices=["Naive Bayes", "Logistic Regression"],
            value="Logistic Regression",
            label="Model",
        ),
    ],
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="📱 SMS Spam Detector",
    description="Type or paste an SMS message to check if it's spam or legitimate (ham).",
    examples=[
        ["Congratulations! You won a free iPhone. Click here to claim your prize now.", "Logistic Regression"],
        ["Hey, are you coming to the party tonight?", "Logistic Regression"],
        ["URGENT! Your account has been compromised. Send your password to verify.", "Logistic Regression"],
        ["I'll be home in 20 minutes, can you start dinner?", "Logistic Regression"],
        ["FREE entry in a weekly competition to win a FA Cup final ticket. Text WED to 87121 now!", "Naive Bayes"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme="soft")
