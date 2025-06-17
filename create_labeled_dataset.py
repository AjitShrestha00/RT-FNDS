import pandas as pd
import os
from datetime import datetime

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Real news data with categories and sources
real_news_data = {
    'title': [
        "NASA's Perseverance Rover Successfully Lands on Mars",
        "New Study Shows Benefits of Regular Exercise",
        "Scientists Discover New Species in Amazon Rainforest",
        "Global Climate Summit Reaches New Agreement",
        "Breakthrough in Quantum Computing Research",
        "World Health Organization Approves New Vaccine",
        "Major Tech Company Announces Renewable Energy Initiative",
        "Archaeologists Uncover Ancient City Remains",
        "New Study Links Diet to Longevity",
        "International Space Station Celebrates 20 Years",
        "COVID-19 Vaccine Shows 95% Effectiveness in Clinical Trials",
        "Renewable Energy Surpasses Coal in US Power Generation",
        "New Species of Deep-Sea Creatures Discovered",
        "Breakthrough in Alzheimer's Research",
        "Global Internet Coverage Initiative Launched"
    ],
    'text': [
        "NASA's Perseverance rover successfully landed on Mars, beginning its mission to search for signs of ancient life and collect samples for future return to Earth.",
        "A comprehensive study published in the Journal of Medicine shows that regular exercise can reduce the risk of heart disease by up to 30%.",
        "A team of scientists has discovered a new species of frog in the Amazon rainforest, highlighting the region's rich biodiversity.",
        "World leaders have reached a new agreement on climate change, setting ambitious targets for reducing carbon emissions.",
        "Researchers have made a significant breakthrough in quantum computing, potentially revolutionizing data processing capabilities.",
        "The World Health Organization has approved a new vaccine that could help prevent millions of cases of disease annually.",
        "A leading technology company has announced plans to power all its operations with 100% renewable energy by 2025.",
        "Archaeologists have uncovered the remains of an ancient city that could provide new insights into early human civilization.",
        "A new study published in Nature shows that a Mediterranean diet is associated with longer life expectancy.",
        "The International Space Station marks its 20th anniversary of continuous human presence in space.",
        "Clinical trials of a new COVID-19 vaccine have shown 95% effectiveness in preventing the disease, according to peer-reviewed research.",
        "For the first time in US history, renewable energy sources have generated more electricity than coal, marking a significant shift in energy production.",
        "Marine biologists have discovered several new species of deep-sea creatures during an expedition in the Pacific Ocean.",
        "Scientists have identified a new approach to treating Alzheimer's disease that shows promising results in early clinical trials.",
        "A global initiative has been launched to provide internet coverage to remote areas using satellite technology."
    ],
    'category': [
        'Science', 'Health', 'Science', 'Environment', 'Technology',
        'Health', 'Technology', 'Science', 'Health', 'Science',
        'Health', 'Environment', 'Science', 'Health', 'Technology'
    ],
    'source': [
        'NASA', 'Journal of Medicine', 'Nature', 'UN Climate Report', 'Science Journal',
        'WHO', 'Tech Company Press Release', 'Archaeological Journal', 'Nature', 'NASA',
        'Medical Journal', 'Energy Department', 'Marine Biology Journal', 'Medical Research', 'Tech News'
    ],
    'date': [
        '2021-02-18', '2023-01-15', '2023-03-20', '2023-04-05', '2023-02-10',
        '2023-01-20', '2023-03-15', '2023-02-28', '2023-03-01', '2023-04-12',
        '2023-01-25', '2023-03-10', '2023-04-01', '2023-02-15', '2023-03-25'
    ],
    'label': [0] * 15,  # 0 for real news
    'verification_status': ['Verified'] * 15
}

# Fake news data with categories and sources
fake_news_data = {
    'title': [
        "Aliens Make First Contact with Earth Government",
        "Scientists Discover Coffee Makes You Immortal",
        "Man Claims to Have Found Unicorn in Backyard",
        "New Study Shows Chocolate Cures All Diseases",
        "Breaking: Dragons Discovered in Remote Mountain",
        "Time Travel Machine Invented by Local Scientist",
        "World's First Flying Car Available for $100",
        "Ancient Mermaid Skeleton Found on Beach",
        "New App Can Read Your Thoughts",
        "Scientists Create Real-Life Invisibility Cloak",
        "5G Networks Found to Control Human Minds",
        "Secret Moon Base Discovered by Amateur Astronomer",
        "New Technology Allows Humans to Breathe Underwater",
        "Government Confirms Existence of Bigfoot",
        "Flying Cars to Be Available at Walmart Next Week"
    ],
    'text': [
        "In a shocking development, extraterrestrial beings have made first contact with world leaders, demanding to speak about intergalactic trade deals.",
        "A groundbreaking study reveals that drinking coffee can make you immortal, with a 100% success rate in laboratory tests.",
        "Local authorities have confirmed the discovery of a real unicorn in a resident's backyard, causing excitement in the scientific community.",
        "Researchers have found that eating chocolate can cure all known diseases, according to a new study published today.",
        "Explorers have discovered a colony of living dragons in a remote mountain range, according to sources.",
        "A local scientist claims to have invented a working time machine, offering demonstrations to the public.",
        "The world's first flying car is now available for just $100, according to an anonymous source.",
        "Beachgoers have discovered what appears to be a mermaid skeleton, sparking debate among marine biologists.",
        "A new smartphone app can read your thoughts and convert them into text messages, developers claim.",
        "Scientists have successfully created a real-life invisibility cloak, available for purchase next month.",
        "A viral social media post claims that 5G networks are being used to control human thoughts and behavior.",
        "An amateur astronomer claims to have discovered a secret base on the dark side of the moon through their backyard telescope.",
        "A new technology allows humans to breathe underwater without any equipment, according to an online article.",
        "The government has allegedly confirmed the existence of Bigfoot after years of investigation.",
        "Walmart has announced they will be selling flying cars next week for the price of a regular car."
    ],
    'category': [
        'Conspiracy', 'Health', 'Paranormal', 'Health', 'Paranormal',
        'Technology', 'Technology', 'Paranormal', 'Technology', 'Technology',
        'Conspiracy', 'Conspiracy', 'Technology', 'Paranormal', 'Technology'
    ],
    'source': [
        'Unverified Blog', 'Fake News Site', 'Social Media', 'Clickbait Site', 'Conspiracy Blog',
        'Fake Tech News', 'Scam Website', 'Fake News Site', 'Fake App Store', 'Fake Science News',
        'Conspiracy Forum', 'Fake News Blog', 'Fake Tech Site', 'Fake News Site', 'Fake Retail News'
    ],
    'date': [
        '2023-04-01', '2023-03-15', '2023-02-28', '2023-03-01', '2023-03-10',
        '2023-02-15', '2023-03-20', '2023-03-05', '2023-03-25', '2023-04-01',
        '2023-03-12', '2023-03-18', '2023-03-22', '2023-03-28', '2023-04-02'
    ],
    'label': [1] * 15,  # 1 for fake news
    'verification_status': ['Unverified'] * 15
}

# Create DataFrames
real_df = pd.DataFrame(real_news_data)
fake_df = pd.DataFrame(fake_news_data)

# Add metadata
real_df['is_fake'] = False
fake_df['is_fake'] = True

# Combine the datasets
combined_df = pd.concat([real_df, fake_df], ignore_index=True)

# Shuffle the data
combined_df = combined_df.sample(frac=1, random_state=42)

# Split into train and test sets (80% train, 20% test)
train_df = combined_df.sample(frac=0.8, random_state=42)
test_df = combined_df.drop(train_df.index)

# Save the datasets
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Create a summary file
summary = {
    'total_samples': len(combined_df),
    'real_news_count': len(real_df),
    'fake_news_count': len(fake_df),
    'training_samples': len(train_df),
    'testing_samples': len(test_df),
    'categories': combined_df['category'].unique().tolist(),
    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Save summary
with open('data/dataset_summary.txt', 'w') as f:
    f.write("Dataset Summary\n")
    f.write("===============\n\n")
    f.write(f"Total Samples: {summary['total_samples']}\n")
    f.write(f"Real News Count: {summary['real_news_count']}\n")
    f.write(f"Fake News Count: {summary['fake_news_count']}\n")
    f.write(f"Training Samples: {summary['training_samples']}\n")
    f.write(f"Testing Samples: {summary['testing_samples']}\n")
    f.write(f"\nCategories: {', '.join(summary['categories'])}\n")
    f.write(f"\nCreated on: {summary['creation_date']}\n")

print("Dataset created successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print("\nSample of training data:")
print(train_df[['title', 'category', 'label', 'verification_status']].head())
print("\nDataset summary saved to data/dataset_summary.txt") 