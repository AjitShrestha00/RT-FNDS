# BERT-based Fake News Detection System

This project implements a fake news detection system using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing model.

## Features

- Uses pre-trained BERT model for text classification
- Supports custom training on your own dataset
- Provides real-time predictions for news articles
- Includes evaluation metrics (accuracy, loss)

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- pandas
- numpy
- scikit-learn
- tqdm

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset with news articles and their labels (0 for real, 1 for fake)
2. Run the main script:
```bash
python fake_news_detector.py
```

## How it Works

The system uses a pre-trained BERT model fine-tuned for binary classification (real vs. fake news). The model processes text through the following steps:

1. Tokenization of input text
2. BERT encoding of the tokens
3. Classification layer to predict real/fake
4. Output prediction with confidence score

## Custom Training

To train the model on your own dataset:

1. Prepare your data in the format of texts and labels
2. Modify the `main()` function in `fake_news_detector.py` with your data
3. Run the script to train and evaluate the model

## Example

```python
detector = FakeNewsDetector()
text = "Your news article here"
prediction = detector.predict(text)
print(f"Prediction: {prediction}")
```

## Note

This is a basic implementation and can be improved by:
- Using a larger dataset for training
- Implementing cross-validation
- Adding more evaluation metrics
- Fine-tuning hyperparameters
- Using a more recent BERT variant (e.g., RoBERTa, DistilBERT) 