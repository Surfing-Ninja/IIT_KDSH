"""
Quick test on 5 examples to validate optimizations.
"""

import pandas as pd
import time
from run import load_dataset, run_batch_evaluation

def main():
    print("\n" + "="*80)
    print("TESTING ON 5 EXAMPLES (Optimized Pipeline)")
    print("="*80)
    
    # Load dataset
    df = load_dataset()
    
    # Check label distribution FIRST
    print("\n" + "="*80)
    print("DATASET LABEL DISTRIBUTION:")
    print("="*80)
    if 'label' in df.columns:
        print(df['label'].value_counts())
        print(f"\nTotal: {len(df)} examples")
        label_map = {'consistent': 1, 'contradict': 0}
        if df['label'].dtype == 'object':
            print("\nMapping labels: 'consistent'=1, 'contradict'=0")
    else:
        print("⚠️  No 'label' column found in dataset")
    print("="*80)
    
    # Take first 5 examples
    test_df = df.head(5)
    print(f"\nTesting on {len(test_df)} examples")
    print(f"Novels: {test_df['novel'].unique().tolist()}")
    if 'label' in test_df.columns:
        print(f"Labels in test: {test_df['label'].tolist()}")
    
    # Run batch evaluation
    start_time = time.time()
    results_df = run_batch_evaluation(test_df)
    elapsed = time.time() - start_time
    
    # Statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total time: {elapsed/60:.2f} minutes ({elapsed/len(test_df):.1f} sec/example)")
    print(f"Predictions: {results_df['prediction'].tolist()}")
    
    if 'label' in results_df.columns:
        accuracy = results_df['correct'].mean()
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {results_df['correct'].sum()}/{len(results_df)}")
    
    # Estimated full runtime
    estimated_full = (elapsed / len(test_df)) * len(df) / 60
    print(f"\nEstimated time for full {len(df)} examples: {estimated_full:.1f} minutes ({estimated_full/60:.1f} hours)")
    
    # Save results
    results_df.to_csv("outputs/test_5_results.csv", index=False)
    print(f"\n✓ Results saved to outputs/test_5_results.csv")

if __name__ == "__main__":
    main()
