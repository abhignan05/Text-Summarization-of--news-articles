from flask import Flask, render_template, request
from summarizer import generate_summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        news_article = request.form['news_article']
        summary = generate_summary(news_article)  # No summary type selection now
        return render_template('result.html', news_article=news_article, summary=summary)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
