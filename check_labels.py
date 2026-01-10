"""
Quick check of dataset label distribution.
"""

import pandas as pd

def check_labels():
    print("\n" + "="*80)
    print("DATASET LABEL ANALYSIS")
    print("="*80)
    
    # Load train.csv
    train_df = pd.read_csv("Dataset/train.csv")
    
    print(f"\nTotal examples: {len(train_df)}")
    print("\nLabel distribution:")
    print(train_df['label'].value_counts())
    
    # Calculate percentages
    label_counts = train_df['label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\nFirst 10 examples:")
    print(train_df[['id', 'book_name', 'label']].head(10).to_string())
    
    # Expected: If always predicting 1 (consistent)
    consistent_count = (train_df['label'] == 'consistent').sum()
    contradict_count = (train_df['label'] == 'contradict').sum()
    
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    print(f"If always predicting 'consistent' (1):")
    print(f"  Accuracy would be: {consistent_count}/{len(train_df)} = {consistent_count/len(train_df):.1%}")
    
    print(f"\nIf always predicting 'contradict' (0):")
    print(f"  Accuracy would be: {contradict_count}/{len(train_df)} = {contradict_count/len(train_df):.1%}")
    
    print("\nRandom guessing (50-50):")
    print(f"  Expected accuracy: ~50%")
    print("="*80)

if __name__ == "__main__":
    check_labels()
