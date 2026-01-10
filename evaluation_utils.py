"""
Utilities for robust parsing, logging, and validation.
Focus: Determinism > novelty, Accuracy > complexity.
"""

import re
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir="outputs/logs"):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized: {log_file}")


def parse_llm_output(output, expected_format="binary"):
    """
    Deterministically parse LLM outputs matching prompts.py specifications.
    
    Handles all 7 prompt output formats:
    1. CONSTRAINT_EXTRACTION_PROMPT → expected_format="json_array"
    2. ESTABLISHMENT_CHECK_PROMPT → expected_format="binary_establishes"
    3. VIOLATION_QUERY_PROMPT → expected_format="plain_text"
    4. VIOLATION_CHECK_PROMPT → expected_format="binary_violates"
    5. REVISION_CHECK_PROMPT → expected_format="binary_revision"
    6. RETRIEVAL_QUALITY_PROMPT → expected_format="text"
    7. SANITY_CHECK_PROMPT → expected_format="text"
    
    Args:
        output: Raw LLM output text from Qwen (temperature=0.0)
        expected_format: Type of output expected
            - "json_array": JSON array of constraint strings
            - "binary_establishes": ESTABLISHES or DOES_NOT_ESTABLISH
            - "binary_violates": VIOLATES or DOES_NOT_VIOLATE
            - "binary_revision": REVISION or VIOLATION
            - "plain_text": Search query string
            - "text": Free-form explanation
            - "binary": Legacy format (maps to specific binary types)
        
    Returns:
        parsed_output: 
            - List[str] for json_array
            - bool for binary_* (True/False/None if ambiguous)
            - str for plain_text/text
    """
    import json
    
    output = output.strip()
    
    # Handle JSON array format (constraint extraction)
    if expected_format == "json_array" or expected_format == "constraints":
        # Remove markdown code blocks if present
        output_clean = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
        output_clean = re.sub(r'^```\s*$', '', output_clean, flags=re.MULTILINE)
        
        # Handle DeepSeek-R1's <think> tags - extract content after </think>
        if '</think>' in output_clean:
            # Get everything after the closing think tag
            parts = output_clean.split('</think>')
            if len(parts) > 1:
                output_clean = parts[-1].strip()
                logging.info("Stripped DeepSeek-R1 reasoning chain")
        
        output_clean = output_clean.strip()
        
        # Try direct JSON parse first
        try:
            parsed = json.loads(output_clean)
            if isinstance(parsed, list):
                return parsed
            else:
                logging.warning(f"JSON parsed but not a list: {type(parsed)}")
        except (json.JSONDecodeError, ValueError) as e:
            # Try to extract JSON array using regex (greedy match to get full array)
            json_match = re.search(r'\[.*\]', output_clean, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        logging.info(f"Extracted JSON using regex: {len(parsed)} items")
                        return parsed
                except Exception as parse_err:
                    logging.warning(f"Regex extracted JSON but parse failed: {parse_err}")
            
            logging.warning(f"JSON parse failed: {e}")
            logging.warning(f"Output (first 200 chars): {output[:200]}")
        
        return []
    
    # Handle binary keyword formats
    elif expected_format == "binary_establishes":
        return _parse_binary_keyword(output, "ESTABLISHES", "DOES_NOT_ESTABLISH")
    
    elif expected_format == "binary_violates":
        return _parse_binary_keyword(output, "VIOLATES", "DOES_NOT_VIOLATE")
    
    elif expected_format == "binary_revision":
        return _parse_binary_keyword(output, "REVISION", "VIOLATION")
    
    # Handle plain text (query generation)
    elif expected_format == "plain_text":
        # Remove markdown code blocks
        text = re.sub(r'^```[a-z]*\s*', '', output, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    # Handle free-form text (explanations)
    elif expected_format == "text":
        return output
    
    # Legacy "binary" format - backward compatibility
    elif expected_format == "binary":
        output_upper = output.upper()
        
        # Check all binary keywords in priority order
        if "VIOLATES" in output_upper and "DOES_NOT_VIOLATE" not in output_upper:
            return "VIOLATES"
        if "DOES_NOT_VIOLATE" in output_upper:
            return "DOES_NOT_VIOLATE"
        if "ESTABLISHES" in output_upper and "DOES_NOT_ESTABLISH" not in output_upper:
            return "ESTABLISHES"
        if "DOES_NOT_ESTABLISH" in output_upper:
            return "DOES_NOT_ESTABLISH"
        if "REVISION" in output_upper and "VIOLATION" not in output_upper:
            return "REVISION"
        if "VIOLATION" in output_upper and "REVISION" not in output_upper:
            return "VIOLATION"
        
        # Fallback: return first word
        first_word = output.split()[0] if output.split() else "UNKNOWN"
        logging.warning(f"Binary parse fallback: {first_word} from '{output[:100]}'")
        return first_word.upper()
    
    else:
        logging.warning(f"Unknown format: {expected_format}")
        return output


def _parse_binary_keyword(text, positive_keyword, negative_keyword):
    """
    Parse binary keyword with strict pattern matching.
    
    Args:
        text: Raw output
        positive_keyword: Keyword for True (e.g., "ESTABLISHES")
        negative_keyword: Keyword for False (e.g., "DOES_NOT_ESTABLISH")
    
    Returns:
        True, False, or None (if ambiguous/not found)
    """
    text_upper = text.upper()
    
    has_positive = positive_keyword.upper() in text_upper
    has_negative = negative_keyword.upper() in text_upper
    
    # Ambiguous if both present
    if has_positive and has_negative:
        logging.warning(f"Ambiguous binary output: both '{positive_keyword}' and '{negative_keyword}' found")
        return None
    
    if has_positive:
        return True
    if has_negative:
        return False
    
    # Neither found
    logging.warning(f"Binary keyword not found in: {text[:100]}")
    return None


def log_step(step_name, input_data, output_data, metadata=None):
    """
    Log a pipeline step with input/output.
    
    Args:
        step_name: Name of the step
        input_data: Input to the step (truncated if long)
        output_data: Output from the step
        metadata: Additional metadata dict
    """
    # Truncate long inputs for readability
    def truncate(text, max_len=200):
        if isinstance(text, str) and len(text) > max_len:
            return text[:max_len] + "..."
        return text
    
    log_msg = f"\n{'='*60}\n[{step_name}]\n"
    log_msg += f"Input: {truncate(str(input_data))}\n"
    log_msg += f"Output: {truncate(str(output_data))}\n"
    
    if metadata:
        log_msg += f"Metadata: {metadata}\n"
    
    logging.info(log_msg)


def sanity_check(constraint, evidence, result):
    """
    Perform sanity checks on constraint extraction and violation detection.
    Simple heuristics to catch obvious errors.
    
    Args:
        constraint: Extracted constraint
        evidence: Supporting evidence text
        result: Detection result (VIOLATES, SUPPORTS, etc.)
        
    Returns:
        is_valid: Boolean indicating if results pass sanity checks
        issues: List of any issues found
    """
    issues = []
    
    # Check constraint is not empty
    if not constraint or len(constraint.strip()) < 5:
        issues.append("Constraint is too short or empty")
    
    # Check evidence is not empty
    if not evidence or len(evidence.strip()) < 10:
        issues.append("Evidence is too short or empty")
    
    # Check result is valid
    valid_results = {
        "VIOLATES", "SUPPORTS", "IRRELEVANT",
        "ESTABLISHES", "CONTRADICTS",
        "REVISION", "VIOLATION",
        "VALID", "INVALID",
        "RELEVANT"
    }
    if result not in valid_results:
        issues.append(f"Invalid result: {result}")
    
    # Check for obvious contradictions
    if result == "VIOLATES" and len(evidence) < 20:
        issues.append("Violation detected but evidence is very short")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logging.warning(f"Sanity check failed: {issues}")
    
    return is_valid, issues


def validate_retrieval_quality(retrieved_chunks, query, min_relevance_score=0.5):
    """
    Validate that retrieved chunks are relevant to the query.
    
    Args:
        retrieved_chunks: List of dicts with 'text' and 'score'
        query: Search query
        min_relevance_score: Minimum acceptable relevance
        
    Returns:
        is_valid: Boolean indicating quality
        avg_score: Average relevance score
    """
    if not retrieved_chunks:
        return False, 0.0
    
    scores = [chunk.get('score', 0.0) for chunk in retrieved_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    is_valid = avg_score >= min_relevance_score
    
    if not is_valid:
        logging.warning(f"Low retrieval quality: avg_score={avg_score:.3f}")
    
    return is_valid, avg_score


def format_evidence(evidence_dict):
    """
    Format evidence dictionary for output CSV.
    
    Args:
        evidence_dict: Dict containing evidence fields
        
    Returns:
        formatted: Human-readable evidence string
    """
    if not evidence_dict:
        return ""
    
    parts = []
    
    if 'constraint' in evidence_dict:
        parts.append(f"Constraint: {evidence_dict['constraint']}")
    
    if 'establishment_text' in evidence_dict:
        text = evidence_dict['establishment_text'][:200]
        parts.append(f"Established: {text}...")
    
    if 'violation_text' in evidence_dict:
        text = evidence_dict['violation_text'][:200]
        parts.append(f"Violation: {text}...")
    
    return " | ".join(parts)


def extract_position_from_chunk(chunk_metadata):
    """
    Extract narrative position from chunk metadata.
    
    Args:
        chunk_metadata: Metadata dict from Pathway
        
    Returns:
        position: Integer position in narrative (0-indexed)
    """
    if isinstance(chunk_metadata, dict):
        return chunk_metadata.get('position', 0)
    return 0


def compute_hybrid_score(dense_score, bm25_score, alpha=0.7):
    """
    Compute hybrid retrieval score.
    
    Args:
        dense_score: Dense embedding similarity
        bm25_score: BM25 lexical score
        alpha: Weight for dense score (1-alpha for BM25)
        
    Returns:
        hybrid_score: Combined score
    """
    return alpha * dense_score + (1 - alpha) * bm25_score
