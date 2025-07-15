from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔁 Load model and tokenizer once
MODEL_PATH = "saved_model"
model, tokenizer = None, None

try:
    print("🚀 Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ Model & tokenizer loaded successfully.")
except Exception as e:
    print("❌ Error loading model/tokenizer:", e)

# 🔍 Text cleaning function
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

# 🔘 Home route (single input if needed)
@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            prediction, confidence = predict_sentiment(review)

    return render_template("index.html", prediction=prediction, confidence=confidence, review=review)


# 📁 Batch upload route
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "POST":
        if 'csvfile' not in request.files:
            return render_template("batch.html", error="No file part found.")

        file = request.files['csvfile']
        if file.filename == "":
            return render_template("batch.html", error="No selected file.")

        if file and file.filename.endswith(".csv"):
            try:
                # Ensure utf-8 decoding for non-Windows generated CSVs
                df = pd.read_csv(file, encoding='utf-8')
                if "review" not in df.columns:
                    return render_template("batch.html", error="CSV must have a 'review' column.")

                results = []
                for i, text in enumerate(df["review"].fillna("").tolist()):
                    sentiment, confidence = predict_sentiment(text)
                    print(f"🧠 Review {i+1}: {text[:40]}... → {sentiment} ({confidence})")
                    results.append({
                        "text": text,
                        "sentiment": sentiment,
                        "confidence": confidence
                    })

                return render_template("batch.html", results=results)

            except Exception as e:
                print("❌ CSV Processing error:", e)
                return render_template("batch.html", error=f"Processing error: {str(e)}")

        return render_template("batch.html", error="Invalid file format. Upload .csv only.")

    return render_template("batch.html")

# 🏁 Start the Flask app
if __name__ == "__main__":
    print("🌐 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
