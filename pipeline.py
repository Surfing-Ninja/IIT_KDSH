"""
End-to-end pipeline orchestration for constraint consistency checking.

STRICT PIPELINE FLOW (NO REORDERING, NO SHORTCUTS):
1. Load models (Qwen, BGE-M3, reranker)
2. Ingest novel via Pathway
3. Extract constraints (Qwen, once)
4. For each constraint:
   a. Find constraint birth using retrieval + LLM
   b. Generate violation search query
   c. Retrieve suspicious chunks after birth
   d. Rerank chunks
   e. Run violation checks
5. Aggregate results
6. Output binary label (0 or 1)

CONSTRAINTS:
- No step may be skipped
- No step may be reordered
- No heuristic shortcuts
- Fully reproducible (temperature=0.0)
"""

from models import load_llm, load_embedder, load_reranker
from pathway_store import PathwayVectorStore
from constraint_engine import check_constraint_consistency
from evaluation_utils import log_step
import config


def load_models():
    """
    STEP 1: Load all required models.
    
    Models loaded:
    - Qwen2.5-14B-Instruct (4-bit NF4, CUDA only)
    - BAAI/bge-m3 (1024-dim embeddings)
    - BAAI/bge-reranker-large (CrossEncoder)
    
    CRITICAL: This step must complete before any other operations.
    
    Returns:
        Dict with keys: 'model', 'tokenizer', 'embedder', 'reranker'
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING MODELS")
    print("="*80)
    
    # Load Qwen LLM (4-bit quantized)
    print("\n[1/3] Loading Qwen2.5-14B-Instruct (4-bit)...")
    model, tokenizer = load_llm()
    print("✓ Qwen loaded")
    
    # Load BGE-M3 embedder
    print("\n[2/3] Loading BAAI/bge-m3 embedder...")
    embedder = load_embedder()
    print("✓ BGE-M3 loaded")
    
    # Load BGE reranker
    print("\n[3/3] Loading BAAI/bge-reranker-large...")
    reranker = load_reranker()
    print("✓ Reranker loaded")
    
    models = {
        'model': model,
        'tokenizer': tokenizer,
        'embedder': embedder,
        'reranker': reranker
    }
    
    print("\n✓ STEP 1 COMPLETE: All models loaded")
    print("="*80)
    
    log_step("load_models", {}, {"status": "success"})
    
    return models


def ingest_novel(novel_text: str, novel_id: str, embedder):
    """
    STEP 2: Ingest novel into Pathway vector store.
    
    Process:
    - Read full novel text (no truncation)
    - Chunk into ~1000-token segments (4000 chars)
    - Assign narrative_position metadata
    - Detect chapters
    - Embed with BGE-M3
    - Index in Pathway
    
    CRITICAL: Must complete before constraint extraction.
    
    Args:
        novel_text: Full novel text
        novel_id: Novel identifier
        embedder: BGE-M3 embedding model
        
    Returns:
        PathwayVectorStore: Initialized and populated vector store
    """
    # Reduced logging for cleaner output
    
    # Initialize Pathway vector store
    vector_store = PathwayVectorStore(embedder)
    
    # Ingest full document
    vector_store.ingest_document(novel_text, novel_id)
    
    # Verify ingestion
    total_chunks = vector_store.get_total_chunks()
    total_chapters = vector_store.get_chapter_count()
    
    # Novel ingested successfully
    
    log_step(
        "ingest_novel",
        {"novel_id": novel_id, "text_length": len(novel_text)},
        {"total_chunks": total_chunks, "total_chapters": total_chapters}
    )
    
    return vector_store


def process_statement(
    vector_store,
    models,
    statement: str,
    novel_id: str
):
    """
    STEPS 3-6: Process statement through constraint checking pipeline.
    
    STEP 3: Extract constraints (Qwen, once)
    STEP 4: For each constraint:
        a. Find constraint birth using retrieval + LLM
        b. Generate violation search query
        c. Retrieve suspicious chunks after birth
        d. Rerank chunks
        e. Run violation checks
    STEP 5: Aggregate results
    STEP 6: Output binary label (0 or 1)
    
    CRITICAL: Steps executed in strict order, no shortcuts.
    
    Args:
        vector_store: PathwayVectorStore with ingested novel
        models: Dict with all loaded models
        statement: Backstory statement to check
        novel_id: Novel identifier for logging
        
    Returns:
        Dict with:
            - prediction: int (0=inconsistent, 1=consistent)
            - constraints: List[str]
            - violations: List[Dict]
            - summary: str
    """
    # Processing constraint consistency check
    
    model = models['model']
    tokenizer = models['tokenizer']
    reranker = models['reranker']
    
    # STEP 3-6: Run full constraint consistency check
    # This function implements the exact flow:
    # 3. Extract constraints
    # 4. For each: birth → query → retrieve → rerank → check
    # 5. Aggregate
    # 6. Binary output
    result = check_constraint_consistency(
        vector_store,
        model,
        tokenizer,
        reranker,
        statement,
        novel_id
    )
    
    # Verify result structure
    assert 'prediction' in result, "Missing 'prediction' in result"
    assert result['prediction'] in [0, 1], "Prediction must be 0 or 1"
    
    # Pipeline completed
    
    log_step(
        "process_statement",
        {"statement": statement[:100], "novel_id": novel_id},
        result
    )
    
    return result


def run_full_pipeline(novel_text: str, statement: str, novel_id: str):
    """
    Execute the complete end-to-end pipeline.
    
    STRICT EXECUTION ORDER:
    1. Load models (Qwen, BGE-M3, reranker)
    2. Ingest novel via Pathway
    3. Extract constraints (Qwen, once)
    4. For each constraint:
       - Find constraint birth using retrieval + LLM
       - Generate violation search query
       - Retrieve suspicious chunks after birth
       - Rerank chunks
       - Run violation checks
    5. Aggregate results
    6. Output binary label (0 or 1)
    
    NO SHORTCUTS:
    - All steps executed in order
    - No caching across different novels
    - No heuristic early stopping (except on first violation)
    - Fully reproducible with temperature=0.0
    
    Args:
        novel_text: Full text of the novel
        statement: Backstory statement to check
        novel_id: Identifier for the novel
        
    Returns:
        Dict with:
            - prediction: int (0=inconsistent, 1=consistent)
            - constraints: List[str] (extracted constraints)
            - violations: List[Dict] (violation evidence if any)
            - summary: str (human-readable result)
            - novel_id: str
    """
    print("\n" + "="*80)
    print("STARTING FULL PIPELINE")
    print("="*80)
    print(f"Novel ID: {novel_id}")
    print(f"Statement: {statement[:100]}...")
    print("="*80)
    
    try:
        # STEP 1: Load models
        models = load_models()
        
        # STEP 2: Ingest novel via Pathway
        vector_store = ingest_novel(novel_text, novel_id, models['embedder'])
        
        # STEPS 3-6: Process statement (extract → check → aggregate → output)
        result = process_statement(vector_store, models, statement, novel_id)
        
        # Add novel_id to result
        result['novel_id'] = novel_id
        
        # Final logging
        log_step(
            "run_full_pipeline",
            {"novel_id": novel_id, "statement": statement[:100]},
            {"prediction": result['prediction']},
            result
        )
        
        return result
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR IN PIPELINE")
        print("="*80)
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        
        # Log error
        log_step(
            "run_full_pipeline",
            {"novel_id": novel_id},
            {"error": str(e)},
            {"traceback": traceback.format_exc()}
        )
        
        # Return error result (default to consistent to avoid false positives)
        return {
            'prediction': 1,
            'constraints': [],
            'violations': [],
            'summary': f"Error: {str(e)}",
            'novel_id': novel_id,
            'error': str(e)
        }
