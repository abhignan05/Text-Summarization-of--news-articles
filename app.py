from flask import Flask, render_template, request
from summarizer import summarize_text
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        text = request.form.get("news_article", "").strip()

        if not text:
            return render_template('predict.html', summary="⚠️ Please enter valid text.", input_text="")

        summary = summarize_text(text)  
        return render_template('predict.html', summary=summary, input_text=text)

    return render_template('predict.html', summary="", input_text="")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port)