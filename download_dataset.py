import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Make sure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Paths to Kaggle CSVs (assume already downloaded and unzipped)
fake_path = 'data/Fake.csv'
real_path = 'data/True.csv'

# Load the datasets
fake_news = pd.read_csv(fake_path)
real_news = pd.read_csv(real_path)

# Add labels
fake_news['label'] = 1  # 1 for fake
real_news['label'] = 0  # 0 for real

# Combine the datasets
combined_data = pd.concat([fake_news, real_news], ignore_index=True)

# Shuffle the data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Optionally, keep only the columns we need
combined_data = combined_data[['title', 'text', 'label']]

# Split into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# Save the processed datasets
train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)

print(f"Dataset prepared successfully!")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")
print("\nSample of training data:")
print(train_data.head()) 