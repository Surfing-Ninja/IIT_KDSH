"""
Example usage of Pathway integration for novel consistency checking.

This script demonstrates:
1. Loading models (Qwen, BGE, reranker)
2. Using Pathway to ingest novels
3. Running consistency checks with all features
"""

from pathway_integration import PathwayPipeline
from models import load_qwen_model, load_bge_embedder, load_reranker
from constraint_engine import check_constraint_consistency
from evaluation import setup_evaluation_logging
import config


def main():
    """
    Main example demonstrating Pathway integration.
    """
    print("\n" + "="*80)
    print("PATHWAY INTEGRATION EXAMPLE")
    print("="*80)
    
    # Step 1: Load models
    print("\n[1/4] Loading models...")
    model, tokenizer = load_qwen_model()
    embedder = load_bge_embedder()
    reranker = load_reranker()
    print("✓ Models loaded")
    
    # Step 2: Create Pathway vector store from local folder
    print("\n[2/4] Initializing Pathway vector store...")
    vector_store = PathwayPipeline.create_from_local_folder(
        folder_path="Dataset/Books",
        embedder="BAAI/bge-m3"
    )
    print("✓ Pathway store ready")
    
    # Step 3: Setup logging
    print("\n[3/4] Setting up logging...")
    logger = setup_evaluation_logging("outputs/logs")
    print("✓ Logging configured")
    
    # Step 4: Run consistency check
    print("\n[4/4] Running consistency check...")
    
    test_statement = "Alice was initially imprisoned for refusing to betray her beliefs"
    novel_id = "The Count of Monte Cristo"
    
    result = check_constraint_consistency(
        vector_store=vector_store,
        model=model,
        tokenizer=tokenizer,
        reranker=reranker,
        statement=test_statement,
        novel_id=novel_id,
        enable_refinement=True,
        max_refinement_attempts=2,
        logger=logger
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Statement: {test_statement}")
    print(f"Novel: {novel_id}")
    print(f"Prediction: {result['prediction']} ({'Inconsistent' if result['prediction'] == 0 else 'Consistent'})")
    print(f"Constraints found: {len(result['constraints'])}")
    
    if result['constraints']:
        print("\nExtracted Constraints:")
        for i, c in enumerate(result['constraints'], 1):
            print(f"  {i}. [{c['type'].upper()}] {c['constraint']}")
    
    if result['violations']:
        print("\nViolations Found:")
        for i, v in enumerate(result['violations'], 1):
            print(f"  {i}. Constraint: {v['constraint'][:60]}...")
            print(f"     Established at: position {v['established_at']}")
            print(f"     Violated at: position {v['violation_position']}")
    
    print("\n" + "="*80)
    print("✓ Example completed successfully")
    print("="*80)


def example_gdrive_integration():
    """
    Example showing Google Drive integration (requires credentials).
    """
    print("\n" + "="*80)
    print("GOOGLE DRIVE INTEGRATION EXAMPLE")
    print("="*80)
    
    # Note: Requires Google Drive credentials JSON file
    print("\nTo use Google Drive:")
    print("1. Create a service account in Google Cloud Console")
    print("2. Download credentials.json")
    print("3. Share your Drive folder with the service account email")
    print("4. Get the folder ID from the Drive URL")
    
    # Example code (commented out)
    """
    from pathway_integration import PathwayPipeline
    
    vector_store = PathwayPipeline.create_from_gdrive(
        folder_id="your_folder_id_here",
        embedder="BAAI/bge-m3"
    )
    
    # Then use vector_store as normal
    result = check_constraint_consistency(...)
    """
    
    print("\n✓ See pathway_integration.py for implementation details")


def example_batch_processing():
    """
    Example showing batch processing of multiple statements.
    """
    print("\n" + "="*80)
    print("BATCH PROCESSING EXAMPLE")
    print("="*80)
    
    import pandas as pd
    
    # Load dataset
    df = pd.read_csv("Dataset/train.csv")
    print(f"\nLoaded {len(df)} examples")
    
    # Load models once
    print("\nLoading models...")
    model, tokenizer = load_qwen_model()
    embedder = load_bge_embedder()
    reranker = load_reranker()
    
    # Create Pathway store once
    print("Initializing Pathway store...")
    vector_store = PathwayPipeline.create_from_local_folder(
        folder_path="Dataset/Books",
        embedder="BAAI/bge-m3"
    )
    
    # Process each example
    results = []
    for idx, row in df.iterrows():
        print(f"\nProcessing {idx+1}/{len(df)}: {row['novel']}")
        
        result = check_constraint_consistency(
            vector_store=vector_store,
            model=model,
            tokenizer=tokenizer,
            reranker=reranker,
            statement=row['statement'],
            novel_id=row['novel']
        )
        
        results.append({
            'novel': row['novel'],
            'statement': row['statement'],
            'prediction': result['prediction'],
            'num_constraints': len(result['constraints']),
            'num_violations': len(result['violations'])
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/batch_results.csv", index=False)
    print(f"\n✓ Results saved to outputs/batch_results.csv")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to try other examples:
    # example_gdrive_integration()
    # example_batch_processing()
