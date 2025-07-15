# predictor.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# 🔁 Load model and tokenizer
MODEL_PATH = "saved_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("🚀 Loading tokenizer and model from predictor.py...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ Model & tokenizer loaded successfully in predictor.py.")
except Exception as e:
    print("❌ Error loading model/tokenizer in predictor.py:", e)

# 🔍 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 🔮 Predict function
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
