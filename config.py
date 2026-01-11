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
RETRIEVAL_TOP_K = 12  # Reduced from 20 to avoid over-retrieval
RERANK_TOP_K = 4  # Reduced from 5 to focus on best matches

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
MAX_CONSTRAINTS_PER_STATEMENT = 3  # Reduced from 5 - more constraints = more false positives

# Violatable constraint types (CRITICAL FOR ACCURACY)
VIOLATABLE_TYPES = {"PROHIBITION", "BELIEF"}  # Only these can have violations
# MOTIVATION, BACKGROUND_FACT, FEAR - skip violation search

# NLI threshold configuration (OPTIMIZED FOR MNLI ON LONG TEXT)
NLI_STRONG_THRESHOLD = 0.75  # High confidence contradiction
NLI_WEAK_THRESHOLD = 0.60  # Potential contradiction
NLI_IGNORE_BELOW = 0.60  # Below this, ignore

# Violation scoring weights (SOFT AGGREGATION)
SCORE_WEIGHT_NLI = 0.5
SCORE_WEIGHT_RERANK = 0.3
SCORE_WEIGHT_LLM = 0.2

# Final decision threshold (CALIBRATED)
VIOLATION_DECISION_THRESHOLD = 0.7  # Score >= 0.7 = violation

# Legacy binary mode (for comparison)
USE_BINARY_MODE = False  # Set True to use old "any violation = 0" logic
USE_SOFT_SCORING = True  # Set False to disable soft aggregation

# Minimum confidence for violation detection (legacy)
VIOLATION_CONFIDENCE_THRESHOLD = 0.8

# Maximum search iterations for violation detection
MAX_VIOLATION_SEARCH_ITERATIONS = 1  # Disabled refinement - not helping, triples runtime

# Minimum retrieval score to auto-accept establishment (bypass LLM check)
ESTABLISHMENT_AUTO_ACCEPT_SCORE = 0.65  # High semantic match = likely established
