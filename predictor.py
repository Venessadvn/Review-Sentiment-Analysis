# predictor.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# ✅ Load model and tokenizer from Hugging Face Hub
MODEL_NAME = "SreyaDvn/sentiment-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("🚀 Loading tokenizer and model from Hugging Face Hub...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("✅ Model & tokenizer loaded successfully from Hugging Face.")
except Exception as e:
    print("❌ Error loading model/tokenizer:", e)

# 🔍 Text cleaner
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 🔮 Sentiment prediction
def predict_sentiment(text):
    try:
        cleaned = clean_text(text)
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_class = probs.argmax()
        sentiment = "Positive 👍🏻" if pred_class == 1 else "Negative 👎🏻"
        confidence = f"{probs[pred_class] * 100:.2f}%"
        return sentiment, confidence
    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error", "0.00%"
