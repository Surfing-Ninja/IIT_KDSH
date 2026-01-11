"""
Configuration file for all model and retrieval settings.
"""

# ============================================================================
# MODEL CONSTRAINTS (NON-NEGOTIABLE)
# ============================================================================

# Qwen Model Configuration - ONLY LLM allowed
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
QWEN_QUANTIZATION = "4bit"  # NF4 quantization for T4 compatibility
QWEN_COMPUTE_DTYPE = "float16"

# Embedding Model Configuration - ONLY embedding model allowed
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 dimension

# Reranker Model Configuration - ONLY reranker allowed
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Optimized for narrative coherence (~1000 tokens per chunk)
# Token estimation: ~4 chars/token → 1000 tokens ≈ 4000 chars
CHUNK_SIZE = 4000  # characters per chunk (~1000 tokens)
CHUNK_OVERLAP = 400  # overlap (~100 tokens) to preserve context boundaries

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Retrieval parameters (OPTIMIZED FOR ACCURACY)
RETRIEVAL_TOP_K = 60  # INCREASED from 40 - more candidates for narrative
RERANK_TOP_K = 12  # INCREASED from 5 - check more candidates
RERANK_MIN_SCORE = 0.0  # REMOVED filtering - don't filter contradictions

# BM25 hybrid retrieval
BM25_ENABLED = True  # Lexical fallback
BM25_WEIGHT = 0.3  # Hybrid fusion weight (0.7 dense + 0.3 BM25)

# Position filtering for constraint persistence
POSITION_FILTER_ENABLED = True
POSITION_FILTER_MODE = "after"  # Only retrieve chunks AFTER constraint establishment

# ============================================================================
# PATHWAY CONFIGURATION
# ============================================================================

PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8754
PATHWAY_PERSISTENCE_PATH = "./pathway_data"

# ============================================================================
# GENERATION CONFIGURATION (DETERMINISTIC)
# ============================================================================

# Qwen generation parameters - temperature=0.0 for determinism
MAX_NEW_TOKENS = 1024  # Increased for DeepSeek-R1 reasoning chains
TEMPERATURE = 0.01  # Changed from 0.0 to avoid generation issues
TOP_P = 0.95  # Nucleus sampling
DO_SAMPLE = True  # Required for temperature > 0

# ============================================================================
# CONSTRAINT ENGINE CONFIGURATION
# ============================================================================

# Max constraints to extract per statement (CAPPED FOR ACCURACY)
MAX_CONSTRAINTS_PER_STATEMENT = 5  # INCREASED from 2 - was too restrictive

# Violatable constraint types (ALL types are violatable)
VIOLATABLE_TYPES = {"PROHIBITION", "BELIEF", "MOTIVATION", "FEAR", "BACKGROUND_FACT"}
# Dataset treats background facts as contradictory when changed - must include them

# NLI threshold configuration (OPTIMIZED FOR MNLI ON LONG TEXT)
NLI_CONTRADICTION_THRESHOLD = 0.25  # LOWERED to 0.25 - narrative contradictions score lower!
NLI_ENTAILMENT_THRESHOLD = 0.25  # LOWERED from 0.60 - more lenient

# Violation scoring weights (SOFT AGGREGATION)
SCORE_WEIGHT_NLI = 0.5
SCORE_WEIGHT_RERANK = 0.3
SCORE_WEIGHT_LLM = 0.2

# Final decision threshold (CALIBRATED)
# Final decision threshold (unused in strict voting mode)
VIOLATION_DECISION_THRESHOLD = 0.5  # kept for compatibility

# Legacy binary mode (for comparison)
USE_BINARY_MODE = True  # Default to binary mode, but pipeline will enforce strict voting
USE_SOFT_SCORING = False  # Disable soft scoring - not used with strict gating

# Minimum confidence for violation detection (legacy)
VIOLATION_CONFIDENCE_THRESHOLD = 0.8

# Maximum search iterations for violation detection
MAX_VIOLATION_SEARCH_ITERATIONS = 1  # Keep refinement disabled to limit runtime

# Position window (how far after establishment we consider violations)
VIOLATION_POSITION_WINDOW = 999999  # UNLIMITED - check entire novel (contradictions can be anywhere)

# Minimum number of confirmed violations required to mark statement inconsistent
MIN_CONFIRMED_VIOLATIONS = 1  # Single violation is enough (most contradictions are single-event)

# Minimum retrieval score to auto-accept establishment (bypass LLM check)
ESTABLISHMENT_AUTO_ACCEPT_SCORE = 0.60  # Slightly lowered from 0.65

# ============================================================================
# CONSTRAINT-TYPE-SPECIFIC VALIDATION
# ============================================================================

# Keywords required for MOTIVATION constraint violations (mental state language)
MOTIVATION_KEYWORDS = {
    "wanted", "want", "wants", "desire", "desires", "desired", "intend", "intends",
    "intended", "intention", "goal", "goals", "aim", "aims", "aimed", "seek", "seeks",
    "sought", "hope", "hopes", "hoped", "plan", "plans", "planned", "purpose",
    "motivation", "motivated", "drive", "driven", "aspire", "aspires", "aspired"
}

# Keywords required for BELIEF constraint violations (belief markers)
BELIEF_KEYWORDS = {
    "believe", "believes", "believed", "think", "thinks", "thought", "feel", "feels",
    "felt", "know", "knows", "knew", "assume", "assumes", "assumed", "suppose",
    "supposes", "supposed", "consider", "considers", "considered", "opinion",
    "convinced", "certain", "sure", "doubt", "doubts", "doubted", "suspect",
    "suspects", "suspected", "trust", "trusts", "trusted", "faith"
}

# Keywords required for PROHIBITION constraint violations (action occurring)
PROHIBITION_ACTION_KEYWORDS = {
    "did", "does", "done", "went", "go", "goes", "made", "make", "makes",
    "took", "take", "takes", "gave", "give", "gives", "came", "come", "comes",
    "left", "leave", "leaves", "entered", "enter", "enters", "opened", "open",
    "opens", "closed", "close", "closes", "started", "start", "starts",
    "finished", "finish", "finishes", "completed", "complete", "completes"
}
