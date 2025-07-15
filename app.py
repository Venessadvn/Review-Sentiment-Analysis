from flask import Flask, render_template, request
import pandas as pd
from predictor import predict_sentiment
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)

# 🔘 Single review input route
@app.route("/")
def root():
    return redirect("/sentiment-review/single")
@app.route("/sentiment-review/single", methods=["GET", "POST"])
def single_review():
    prediction = None
    confidence = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            prediction, confidence = predict_sentiment(review)

    return render_template("index.html", prediction=prediction, confidence=confidence, review=review)

# 📁 Batch upload route
@app.route("/sentiment-review/batch", methods=["GET", "POST"])
def batch_review():
    if request.method == "POST":
        if 'csvfile' not in request.files:
            return render_template("batch.html", error="No file part found.")

        file = request.files['csvfile']
        if file.filename == "":
            return render_template("batch.html", error="No selected file.")

        if file and file.filename.endswith(".csv"):
            try:
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
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
