from flask import Flask, render_template, request, jsonify
from fake_news_detector import FakeNewsDetector
import torch

app = Flask(__name__)

# Initialize the detector
detector = FakeNewsDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the news text from the request
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Make prediction
        prediction = detector.predict(news_text)
        
        return jsonify({
            'prediction': prediction,
            'text': news_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 