import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FakeNewsDetector:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)
        
        # Load the model if it exists
        if os.path.exists('model/fake_news_model.pt'):
            self.model.load_state_dict(torch.load('model/fake_news_model.pt'))
            print("Loaded pre-trained model")

    def prepare_data(self, data_path, batch_size=8):
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Combine title and text
        df['full_text'] = df['title'] + ' ' + df['text']
        
        # Create dataset
        dataset = NewsDataset(df['full_text'].values, df['label'].values, self.tokenizer)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Calculate training accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += len(labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Training accuracy: {train_accuracy:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            print(f'Validation accuracy: {val_accuracy:.4f}')
            
            # Save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if not os.path.exists('model'):
                    os.makedirs('model')
                torch.save(self.model.state_dict(), 'model/fake_news_model.pt')
                print("Saved best model!")

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += len(labels)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
        return "Fake" if predictions.item() == 1 else "Real"

def main():
    # Initialize the detector
    detector = FakeNewsDetector()
    
    # Prepare the data
    train_loader = detector.prepare_data('data/train.csv')
    val_loader = detector.prepare_data('data/test.csv')
    
    # Train the model
    detector.train(train_loader, val_loader)
    
    # Test the model
    test_texts = [
        "NASA's Perseverance rover successfully landed on Mars, beginning its mission to search for signs of ancient life.",
        "Aliens make first contact with Earth government, demand to speak with world leaders.",
        "New study shows that regular exercise can reduce the risk of heart disease by up to 30%.",
        "Scientists discover that drinking coffee makes you immortal with 100% success rate."
    ]
    
    print("\nTesting the model:")
    for text in test_texts:
        prediction = detector.predict(text)
        print(f"\nText: {text[:100]}...")
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main() 