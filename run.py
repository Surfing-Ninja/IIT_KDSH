"""
Main entrypoint for batch processing the dataset.
Runs the full pipeline on all novels and generates results.csv.

Usage:
    python run.py
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from pipeline import load_models, ingest_novel, process_statement
from evaluation_utils import setup_logging, format_evidence


# Global models cache
MODELS = None


def load_dataset(data_dir="Dataset"):
    """
    Load the dataset with novels and statements.
    Expected format: CSV with columns [novel, statement, label (optional)]
    
    Args:
        data_dir: Directory containing train.csv, test.csv, and Books/
        
    Returns:
        df: DataFrame with examples
    """
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    # Try to load train.csv first, then test.csv
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        print(f"Loaded train.csv: {len(df)} examples")
    elif os.path.exists(test_path):
        df = pd.read_csv(test_path)
        print(f"Loaded test.csv: {len(df)} examples")
    else:
        raise FileNotFoundError(f"No train.csv or test.csv found in {data_dir}")
    
    # Validate columns and map to expected names
    print(f"Original columns: {df.columns.tolist()}")
    
    # Map actual column names to expected names
    if 'book_name' in df.columns and 'content' in df.columns:
        df = df.rename(columns={'book_name': 'novel', 'content': 'statement'})
        print("Mapped 'book_name' → 'novel' and 'content' → 'statement'")
    
    required_cols = ['novel', 'statement']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"Final columns: {df.columns.tolist()}")
    
    return df


def load_novel_text(novel_name: str, books_dir="Dataset/Books"):
    """
    Load the full text of a novel.
    
    Args:
        novel_name: Name of the novel (e.g., "The Count of Monte Cristo")
        books_dir: Directory containing novel text files
        
    Returns:
        text: Full novel text
    """
    # Try exact match first
    novel_path = os.path.join(books_dir, f"{novel_name}.txt")
    
    if not os.path.exists(novel_path):
        # Try case-insensitive match
        for filename in os.listdir(books_dir):
            if filename.lower() == f"{novel_name.lower()}.txt":
                novel_path = os.path.join(books_dir, filename)
                break
    
    if not os.path.exists(novel_path):
        raise FileNotFoundError(f"Novel not found: {novel_path}")
    
    with open(novel_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def run_batch_evaluation(df: pd.DataFrame, output_path="outputs/results.csv"):
    """
    Run evaluation on all examples in the dataset.
    
    Args:
        df: DataFrame with examples
        output_path: Path to save results
        
    Returns:
        results_df: DataFrame with predictions and evidence
    """
    global MODELS
    
    print("\n" + "="*60)
    print("RUNNING BATCH EVALUATION")
    print("="*60)
    
    # Load models once
    if MODELS is None:
        MODELS = load_models()
    
    results = []
    
    # Group statements by novel to avoid re-embedding
    print(f"\nGrouping statements by novel to optimize embedding...")
    df_grouped = df.groupby('novel')
    total_novels = len(df_grouped)
    print(f"Found {total_novels} unique novels, {len(df)} total statements")
    
    # Cache for vector stores
    vector_store_cache = {}
    
    # Process each example with progress bar
    pbar = tqdm(df.iterrows(), total=len(df), desc="Processing examples", ncols=100)
    for idx, row in pbar:
        novel_name = row['novel']
        statement = row['statement']
        
        # Update progress bar description
        pbar.set_postfix({"Novel": novel_name[:20], "Example": f"{idx+1}/{len(df)}"})
        
        try:
            # Check if we already have this novel's embeddings
            if novel_name not in vector_store_cache:
                # Load novel text and create vector store (only once per novel)
                novel_text = load_novel_text(novel_name)
                vector_store = ingest_novel(novel_text, novel_name, MODELS['embedder'])
                vector_store_cache[novel_name] = vector_store
                pbar.write(f"✓ Cached embeddings for '{novel_name}'")
            else:
                # Reuse existing vector store
                vector_store = vector_store_cache[novel_name]
            
            # Process statement
            result_dict = process_statement(
                vector_store,
                MODELS,
                statement,
                novel_name  # Changed from novel_text to novel_name (novel_id)
            )
            
            # Extract prediction and create evidence summary
            prediction = result_dict['prediction']
            evidence = {
                'summary': result_dict.get('summary', ''),
                'constraints': result_dict.get('constraints', []),
                'violations': result_dict.get('violations', [])
            }
            
            # Format results
            result = {
                'novel': novel_name,
                'statement': statement,
                'prediction': prediction,
                'evidence': format_evidence(evidence)
            }
            
            # Add ground truth if available
            if 'label' in row:
                # Map string labels to numeric (consistent=1, contradict=0)
                label_str = row['label']
                label_numeric = 1 if label_str == 'consistent' else 0
                result['label'] = label_numeric
                result['label_str'] = label_str
                correct = (prediction == label_numeric)
                result['correct'] = correct
            
            results.append(result)
            
        except Exception as e:
            pbar.write(f"\n❌ ERROR on example {idx+1}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            
            # Add error result
            result = {
                'novel': novel_name,
                'statement': statement,
                'prediction': 1,  # Default to consistent on error
                'evidence': f"ERROR: {str(e)}"
            }
            if 'label' in row:
                result['label'] = row['label']
                result['correct'] = False
            
            results.append(result)
    
    # Close progress bar
    pbar.close()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute accuracy if labels available
    if 'label' in results_df.columns:
        accuracy = results_df['correct'].mean()
        print(f"\n{'='*60}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"  Total examples: {len(results_df)}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Correct: {results_df['correct'].sum()}/{len(results_df)}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"  Total examples processed: {len(results_df)}")
        print(f"{'='*60}")
    
    return results_df


def main():
    """
    Main execution function.
    """
    print("\n" + "#"*60)
    print("CONSTRAINT CONSISTENCY CHECKER")
    print("Competition-grade NLP system")
    print("#"*60)
    
    # Setup logging
    setup_logging()
    
    # Load dataset
    df = load_dataset()
    
    # Run evaluation
    results_df = run_batch_evaluation(df)
    
    # Save results
    output_path = "outputs/results.csv"
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED: {output_path}")
    print(f"{'='*60}")
    print(f"Total examples: {len(results_df)}")
    print(f"Results preview:")
    print(results_df.head())
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
