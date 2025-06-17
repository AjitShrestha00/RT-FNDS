import pandas as pd
import os

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Sample real news data
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
        "International Space Station Celebrates 20 Years"
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
        "The International Space Station marks its 20th anniversary of continuous human presence in space."
    ],
    'label': [0] * 10  # 0 for real news
}

# Sample fake news data
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
        "Scientists Create Real-Life Invisibility Cloak"
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
        "Scientists have successfully created a real-life invisibility cloak, available for purchase next month."
    ],
    'label': [1] * 10  # 1 for fake news
}

# Create DataFrames
real_df = pd.DataFrame(real_news_data)
fake_df = pd.DataFrame(fake_news_data)

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

print("Dataset created successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print("\nSample of training data:")
print(train_df[['title', 'label']].head()) 