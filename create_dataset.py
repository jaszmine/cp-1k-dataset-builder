import pandas as pd
from datasets import load_dataset
import re
from collections import defaultdict
import random
import json

'''
This script will download the dataset, filter it, classify with keywrods,
and create the files needed for Label Studio import. 
'''

# Define your categories and distribution
categories = {
    'not_relevant': "General tweets that don't relate to disasters or emergencies",
    'auto_accident': "Car crashes, vehicle collisions, traffic accidents",
    'fire': "Fires, wildfires, building fires, structure fires, explosions",
    'flood': "Flooding, flash floods, water inundation",
    'earthquake': "Earthquakes, tremors, seismic activity",
    'severe_storm': "Severe thunderstorms, hail, lightning, windstorms",
    'shooting': "Mass shootings, gun violence, active shooter situations",
    'tornado': "Tornadoes, funnel clouds, twisters",
    'hurricane': "Hurricanes",
    'extreme_heat': "Heat waves, extreme temperatures, droughts",
    'tropical_storm': "Tropical storms, tropical cyclones, monsoons, typhoons",
    'other_disaster': "Other disasters like avalanches, landslides, volcanic eruptions, tsunamis"
}

ideal_distribution_1000 = {
    'not_relevant': 320,      # 32% - 320 tweets
    'auto_accident': 110,     # 11% - 110 tweets  
    'fire': 100,              # 10% - 100 tweets
    'flood': 100,             # 10% - 100 tweets
    'severe_storm': 90,       # 9% - 90 tweets
    'earthquake': 80,         # 8% - 80 tweets
    'shooting': 70,           # 7% - 70 tweets
    'tornado': 40,            # 4% - 40 tweets
    'hurricane': 30,          # 3% - 30 tweets
    'extreme_heat': 30,       # 3% - 30 tweets
    'tropical_storm': 20,     # 2% - 20 tweets
    'other_disaster': 10      # 1% - 10 tweets
}

def main():
    print("Loading dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("alpindale/two-million-bluesky-posts", split="train")
    df = dataset.to_pandas()
    
    print(f"Loaded {len(df)} total posts")
    
    # Filter for English posts
    print("Filtering for English posts...")
    def is_english(text):
        if not isinstance(text, str):
            return False
        # Simple heuristic - adjust as needed
        return bool(re.match(r'^[a-zA-Z0-9\s\.,!?;:\'"-]+$', text[:100]))
    
    english_df = df[df['text'].apply(is_english)].copy()
    print(f"Found {len(english_df)} English posts")
    
    # Remove duplicates
    print("Removing duplicates...")
    english_df = english_df.drop_duplicates(subset=['text'])
    print(f"After removing duplicates: {len(english_df)} posts")
    
    # Define keyword patterns for each category
    print("Classifying posts by keywords...")
    keyword_patterns = {
        'auto_accident': r'\b(crash|accident|collision|wreck|car accident|vehicle accident|traffic accident|car crash|road accident)\b',
        'fire': r'\b(fire|blaze|wildfire|burning|explosion|ablaze|arson|smoke|flames)\b',
        'flood': r'\b(flood|flooding|inundation|deluge|flash flood|water level|submerged)\b',
        'earthquake': r'\b(earthquake|tremor|seismic|magnitude|epicenter|aftershock|quake)\b',
        'severe_storm': r'\b(storm|thunderstorm|hail|lightning|windstorm|gale|squall|downpour)\b',
        'shooting': r'\b(shooting|gunfire|active shooter|mass shooting|gun violence|shot|fired)\b',
        'tornado': r'\b(tornado|funnel cloud|twister|cyclone|supercell)\b',
        'hurricane': r'\b(hurricane|cyclone|storm surge|eyewall)\b',
        'extreme_heat': r'\b(heat wave|extreme heat|drought|scorching|heatstroke|temperature record)\b',
        'tropical_storm': r'\b(tropical storm|monsoon|typhoon|tropical depression)\b',
        'other_disaster': r'\b(avalanche|landslide|volcano|tsunami|eruption|mudslide|disaster|emergency)\b'
    }
    
    def classify_text(text):
        if not isinstance(text, str):
            return 'not_relevant'
        
        text_lower = text.lower()
        
        for category, pattern in keyword_patterns.items():
            if re.search(pattern, text_lower):
                return category
        
        return 'not_relevant'
    
    # Apply initial classification
    english_df['predicted_label'] = english_df['text'].apply(classify_text)
    
    # Show initial distribution
    print("\nInitial classification distribution:")
    print(english_df['predicted_label'].value_counts())
    
    # Sample according to distribution
    print("\nSampling according to target distribution...")
    sampled_data = []
    
    for category, count in ideal_distribution_1000.items():
        category_posts = english_df[english_df['predicted_label'] == category]
        
        if len(category_posts) >= count:
            sampled = category_posts.sample(n=count, random_state=42)
            print(f"✓ {category}: sampled {count} posts")
        else:
            # If not enough posts, take all available and show warning
            sampled = category_posts
            print(f"⚠ {category}: only found {len(category_posts)} posts (needed {count})")
        
        sampled_data.append(sampled)
    
    sampled_df = pd.concat(sampled_data, ignore_index=True)
    
    # Create Label Studio import format
    print("\nCreating Label Studio import file...")
    tasks = []
    
    for idx, row in sampled_df.iterrows():
        task = {
            "data": {
                "text": row['text'],
                "id": f"post_{idx}"
            },
            "predictions": [{
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices",
                        "value": {
                            "choices": [row['predicted_label']]
                        }
                    }
                ]
            }]
        }
        tasks.append(task)
    
    # Save files
    with open('label_studio_import.json', 'w') as f:
        json.dump(tasks, f, indent=2)
    
    sampled_df[['text', 'predicted_label']].to_csv('pre_labeled_dataset.csv', index=False)
    
    print(f"\n✅ Successfully created dataset with {len(sampled_df)} posts")
    print("📁 Files created:")
    print("   - label_studio_import.json (for Label Studio)")
    print("   - pre_labeled_dataset.csv (backup CSV)")
    
    # Show final sampled distribution
    print("\nFinal sampled distribution:")
    final_dist = sampled_df['predicted_label'].value_counts()
    for category in ideal_distribution_1000.keys():
        count = final_dist.get(category, 0)
        print(f"   {category}: {count} posts")

if __name__ == "__main__":
    main()