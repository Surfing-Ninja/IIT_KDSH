"""
Evaluation utilities for constraint consistency checking.

REQUIREMENTS:
- Deterministic parsing of LLM outputs (no probabilities)
- Defensive handling of malformed responses
- Logging of constraint birth positions, violations, and evidence
- NO chain-of-thought exposure
- NO probability computation

All parsing is strict pattern matching - no fuzzy matching, no heuristics.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_evaluation_logging(log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Set up structured logging for evaluation.
    
    Logs:
    - Constraint birth positions
    - Violated constraints
    - Supporting evidence
    - Parse errors (malformed responses)
    
    Args:
        log_dir: Directory for log files
    
    Returns:
        Logger instance configured for evaluation
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"evaluation_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove existing handlers
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (summary only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Evaluation logging initialized: {log_file}")
    
    return logger


def log_constraint_birth(
    logger: logging.Logger,
    constraint: str,
    position: int,
    evidence_text: str
) -> None:
    """
    Log constraint establishment (birth) position.
    
    Args:
        logger: Logger instance
        constraint: Constraint text
        position: Position in narrative where constraint is established
        evidence_text: Text chunk where constraint is established
    """
    logger.info(f"CONSTRAINT_BIRTH | Position: {position} | Constraint: {constraint[:100]}")
    logger.debug(f"BIRTH_EVIDENCE | {evidence_text[:200]}")


def log_violation(
    logger: logging.Logger,
    constraint: str,
    violation_position: int,
    evidence_text: str,
    is_revision: bool = False
) -> None:
    """
    Log constraint violation (or revision).
    
    Args:
        logger: Logger instance
        constraint: Violated constraint text
        violation_position: Position in narrative where violation occurs
        evidence_text: Text chunk containing violation
        is_revision: True if this is an intentional revision (not a true violation)
    """
    violation_type = "REVISION" if is_revision else "VIOLATION"
    logger.info(f"{violation_type} | Position: {violation_position} | Constraint: {constraint[:100]}")
    logger.debug(f"{violation_type}_EVIDENCE | {evidence_text[:200]}")


def log_parse_error(
    logger: logging.Logger,
    prompt_type: str,
    raw_output: str,
    expected_format: str,
    error: str
) -> None:
    """
    Log malformed LLM response.
    
    Args:
        logger: Logger instance
        prompt_type: Type of prompt (e.g., 'CONSTRAINT_EXTRACTION')
        raw_output: Raw LLM output that failed to parse
        expected_format: Expected output format
        error: Error description
    """
    logger.warning(f"PARSE_ERROR | Prompt: {prompt_type} | Expected: {expected_format}")
    logger.debug(f"RAW_OUTPUT | {raw_output[:500]}")
    logger.debug(f"ERROR_DETAIL | {error}")


def log_evaluation_summary(
    logger: logging.Logger,
    total_constraints: int,
    violated_constraints: int,
    prediction: int,
    execution_time: float
) -> None:
    """
    Log evaluation summary for a statement.
    
    Args:
        logger: Logger instance
        total_constraints: Total number of constraints extracted
        violated_constraints: Number of violated constraints
        prediction: Binary prediction (0 or 1)
        execution_time: Time taken in seconds
    """
    logger.info("="*80)
    logger.info("EVALUATION_SUMMARY")
    logger.info(f"  Total Constraints: {total_constraints}")
    logger.info(f"  Violated Constraints: {violated_constraints}")
    logger.info(f"  Prediction: {prediction} ({'VIOLATION' if prediction == 1 else 'CONSISTENT'})")
    logger.info(f"  Execution Time: {execution_time:.2f}s")
    logger.info("="*80)


# ============================================================================
# DETERMINISTIC PARSING UTILITIES
# ============================================================================

def parse_constraints_with_types(
    raw_output: str,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, str]]:
    """
    Parse constraints with type categorization from LLM output.
    
    Expected format: [{"type": "belief", "constraint": "..."}, ...]
    
    Types: belief, prohibition, motivation, background_fact, fear
    
    Args:
        raw_output: Raw LLM output
        logger: Optional logger for error reporting
    
    Returns:
        List of dicts with 'type' and 'constraint' keys, or [] if parse fails
    """
    if not raw_output or not isinstance(raw_output, str):
        if logger:
            log_parse_error(logger, "CONSTRAINTS_WITH_TYPES", str(raw_output), "JSON array of objects", "Empty or non-string input")
        return []
    
    # Clean markdown code blocks
    cleaned = raw_output.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    try:
        parsed = json.loads(cleaned)
        
        if not isinstance(parsed, list):
            if logger:
                log_parse_error(logger, "CONSTRAINTS_WITH_TYPES", raw_output, "JSON array", f"Parsed as {type(parsed).__name__}, not list")
            return []
        
        # Valid constraint types
        valid_types = {'belief', 'prohibition', 'motivation', 'background_fact', 'fear'}
        
        result = []
        for item in parsed:
            if isinstance(item, dict) and 'constraint' in item:
                constraint_type = item.get('type', 'background_fact')  # Default to background_fact
                constraint_text = str(item['constraint']).strip()
                
                # Validate type
                if constraint_type not in valid_types:
                    if logger:
                        logger.warning(f"Invalid constraint type '{constraint_type}', defaulting to 'background_fact'")
                    constraint_type = 'background_fact'
                
                if constraint_text:
                    result.append({
                        'type': constraint_type,
                        'constraint': constraint_text
                    })
            elif isinstance(item, str):
                # Backward compatibility: plain string → background_fact
                result.append({
                    'type': 'background_fact',
                    'constraint': item.strip()
                })
        
        return result
    
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        if logger:
            log_parse_error(logger, "CONSTRAINTS_WITH_TYPES", raw_output[:200], "JSON array of objects", str(e))
        return []


def parse_plain_text_safe(raw_output: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Defensively parse plain text from LLM output.
    
    Used for query refinement and similar tasks where LLM outputs a single string.
    
    Handles:
    - Direct text output
    - Markdown code blocks
    - Leading/trailing whitespace
    - Empty responses (returns empty string)
    
    Args:
        raw_output: Raw LLM output string
        logger: Optional logger for error reporting
        
    Returns:
        str: Cleaned text or empty string on failure
        
    Examples:
        >>> parse_plain_text_safe("  search for character mentions  ")
        "search for character mentions"
        >>> parse_plain_text_safe("```\\nsearch query\\n```")
        "search query"
        >>> parse_plain_text_safe("")
        ""
    """
    if not raw_output:
        return ""
    
    # Strip markdown code blocks
    cleaned = raw_output.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        # Remove first and last lines
        lines = cleaned.split("\n")
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1])
    
    # Strip whitespace
    cleaned = cleaned.strip()
    
    # If multiline, take first non-empty line (for concise outputs like queries)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if lines:
        return lines[0]
    
    return ""


def parse_json_array_safe(raw_output: str, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Defensively parse JSON array from LLM output.
    
    Handles:
    - Valid JSON arrays
    - Markdown code blocks (```json...```)
    - Malformed JSON (returns empty list)
    - Empty/null responses
    
    Does NOT use fuzzy matching or heuristics.
    
    Args:
        raw_output: Raw LLM output
        logger: Optional logger for error reporting
    
    Returns:
        List of strings, or [] if parse fails
    """
    if not raw_output or not isinstance(raw_output, str):
        if logger:
            log_parse_error(logger, "JSON_ARRAY", str(raw_output), "JSON array", "Empty or non-string input")
        return []
    
    # Clean markdown code blocks
    cleaned = raw_output.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    try:
        parsed = json.loads(cleaned)
        
        if not isinstance(parsed, list):
            if logger:
                log_parse_error(logger, "JSON_ARRAY", raw_output, "JSON array", f"Parsed as {type(parsed).__name__}, not list")
            return []
        
        # Convert all elements to strings
        result = []
        for item in parsed:
            if isinstance(item, str):
                result.append(item.strip())
            elif isinstance(item, dict) and 'constraint' in item:
                # Handle {"constraint": "..."} format
                result.append(str(item['constraint']).strip())
            else:
                # Convert to string
                result.append(str(item).strip())
        
        # Filter out empty strings
        result = [s for s in result if s]
        
        return result
    
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        if logger:
            log_parse_error(logger, "JSON_ARRAY", raw_output[:200], "JSON array", str(e))
        return []


def parse_binary_keyword_safe(
    raw_output: str,
    positive_keyword: str,
    negative_keyword: str,
    logger: Optional[logging.Logger] = None
) -> Optional[bool]:
    """
    Defensively parse binary keyword from LLM output.
    
    Strict pattern matching:
    - Returns True if positive_keyword found
    - Returns False if negative_keyword found
    - Returns None if both found (ambiguous) or neither found
    
    Case-insensitive. NO fuzzy matching.
    
    Args:
        raw_output: Raw LLM output
        positive_keyword: Keyword for True (e.g., "ESTABLISHES")
        negative_keyword: Keyword for False (e.g., "DOES_NOT_ESTABLISH")
        logger: Optional logger for error reporting
    
    Returns:
        True, False, or None (ambiguous/not found)
    """
    if not raw_output or not isinstance(raw_output, str):
        if logger:
            log_parse_error(
                logger,
                "BINARY_KEYWORD",
                str(raw_output),
                f"{positive_keyword}/{negative_keyword}",
                "Empty or non-string input"
            )
        return None
    
    text_upper = raw_output.upper()
    positive_upper = positive_keyword.upper()
    negative_upper = negative_keyword.upper()
    
    has_positive = positive_upper in text_upper
    has_negative = negative_upper in text_upper
    
    # Ambiguous: both present
    if has_positive and has_negative:
        if logger:
            log_parse_error(
                logger,
                "BINARY_KEYWORD",
                raw_output[:200],
                f"{positive_keyword}/{negative_keyword}",
                f"Both keywords found (ambiguous)"
            )
        return None
    
    # Clear positive
    if has_positive:
        return True
    
    # Clear negative
    if has_negative:
        return False
    
    # Neither found
    if logger:
        log_parse_error(
            logger,
            "BINARY_KEYWORD",
            raw_output[:200],
            f"{positive_keyword}/{negative_keyword}",
            "Neither keyword found"
        )
    return None


def parse_plain_text_safe(raw_output: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Defensively parse plain text from LLM output.
    
    Cleans:
    - Markdown code blocks
    - Excessive whitespace
    
    Args:
        raw_output: Raw LLM output
        logger: Optional logger (unused, for consistency)
    
    Returns:
        Cleaned text string
    """
    if not raw_output or not isinstance(raw_output, str):
        return ""
    
    # Remove markdown code blocks
    text = re.sub(r'^```[a-z]*\s*', '', raw_output, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    
    # Collapse excessive newlines (max 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Trim
    return text.strip()


# ============================================================================
# EVALUATION RESULT STRUCTURES
# ============================================================================

def create_constraint_record(
    constraint: str,
    birth_position: int,
    birth_evidence: str,
    violated: bool = False,
    violation_position: Optional[int] = None,
    violation_evidence: Optional[str] = None,
    is_revision: bool = False
) -> Dict[str, Any]:
    """
    Create structured record for a constraint.
    
    Args:
        constraint: Constraint text
        birth_position: Position where constraint is established
        birth_evidence: Text chunk establishing constraint
        violated: Whether constraint is violated
        violation_position: Position of violation (if violated)
        violation_evidence: Text chunk containing violation (if violated)
        is_revision: Whether violation is actually a revision
    
    Returns:
        Dictionary with constraint details
    """
    record = {
        'constraint': constraint,
        'birth_position': birth_position,
        'birth_evidence': birth_evidence[:500] if birth_evidence else "",
        'violated': violated,
    }
    
    if violated:
        record['violation_position'] = violation_position
        record['violation_evidence'] = violation_evidence[:500] if violation_evidence else ""
        record['is_revision'] = is_revision
    
    return record


def create_evaluation_result(
    statement: str,
    prediction: int,
    constraints: List[Dict[str, Any]],
    execution_time: float,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create structured evaluation result.
    
    Args:
        statement: Input statement
        prediction: Binary prediction (0 = consistent, 1 = violation)
        constraints: List of constraint records
        execution_time: Time taken in seconds
        metadata: Optional metadata (novel name, etc.)
    
    Returns:
        Dictionary with complete evaluation result
    """
    violated_constraints = [c for c in constraints if c.get('violated', False)]
    
    result = {
        'statement': statement,
        'prediction': prediction,
        'total_constraints': len(constraints),
        'violated_constraints': len(violated_constraints),
        'constraints': constraints,
        'execution_time': execution_time,
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        result['metadata'] = metadata
    
    return result


def save_evaluation_result(
    result: Dict[str, Any],
    output_path: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save evaluation result to JSON file.
    
    Args:
        result: Evaluation result dictionary
        output_path: Path to save JSON file
        logger: Optional logger
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    if logger:
        logger.info(f"Result saved to {output_path}")


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_constraint_list(
    constraints: List[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, List[str]]:
    """
    Validate extracted constraints for quality.
    
    Checks:
    - Non-empty list
    - All elements are non-empty strings
    - Reasonable length (5-500 characters)
    - No duplicates
    
    Args:
        constraints: List of constraint strings
        logger: Optional logger
    
    Returns:
        (is_valid, issues) tuple
    """
    issues = []
    
    if not constraints:
        issues.append("Empty constraint list")
        return False, issues
    
    # Check all strings
    if not all(isinstance(c, str) for c in constraints):
        issues.append("Non-string constraint found")
    
    # Check non-empty strings
    empty_count = sum(1 for c in constraints if not c.strip())
    if empty_count > 0:
        issues.append(f"{empty_count} empty constraint(s)")
    
    # Check reasonable length (5-500 characters)
    too_short = [c for c in constraints if len(c.strip()) < 5]
    too_long = [c for c in constraints if len(c.strip()) > 500]
    
    if too_short:
        issues.append(f"{len(too_short)} constraint(s) too short (<5 chars)")
    if too_long:
        issues.append(f"{len(too_long)} constraint(s) too long (>500 chars)")
    
    # Check for duplicates
    unique_constraints = set(c.strip() for c in constraints)
    if len(unique_constraints) < len(constraints):
        duplicate_count = len(constraints) - len(unique_constraints)
        issues.append(f"{duplicate_count} duplicate constraint(s)")
    
    is_valid = len(issues) == 0
    
    if logger and not is_valid:
        logger.warning(f"Constraint validation failed: {', '.join(issues)}")
    
    return is_valid, issues


def validate_position(
    position: int,
    max_position: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate narrative position.
    
    Args:
        position: Position to validate
        max_position: Maximum valid position (total chunks)
        logger: Optional logger
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(position, int):
        if logger:
            logger.warning(f"Invalid position type: {type(position).__name__}")
        return False
    
    if position < 0:
        if logger:
            logger.warning(f"Negative position: {position}")
        return False
    
    if position >= max_position:
        if logger:
            logger.warning(f"Position {position} exceeds max {max_position}")
        return False
    
    return True


def validate_evaluation_result(
    result: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, List[str]]:
    """
    Validate evaluation result structure.
    
    Checks:
    - Required fields present
    - Prediction is binary (0 or 1)
    - Constraint counts consistent
    
    Args:
        result: Evaluation result dictionary
        logger: Optional logger
    
    Returns:
        (is_valid, issues) tuple
    """
    issues = []
    
    # Check required fields
    required_fields = ['statement', 'prediction', 'total_constraints', 'violated_constraints', 'constraints']
    for field in required_fields:
        if field not in result:
            issues.append(f"Missing required field: {field}")
    
    # Check prediction is binary
    if 'prediction' in result:
        if result['prediction'] not in [0, 1]:
            issues.append(f"Prediction must be 0 or 1, got {result['prediction']}")
    
    # Check constraint counts
    if 'total_constraints' in result and 'constraints' in result:
        if result['total_constraints'] != len(result['constraints']):
            issues.append(f"Total constraints mismatch: {result['total_constraints']} != {len(result['constraints'])}")
    
    if 'violated_constraints' in result and 'constraints' in result:
        actual_violated = sum(1 for c in result['constraints'] if c.get('violated', False))
        if result['violated_constraints'] != actual_violated:
            issues.append(f"Violated constraints mismatch: {result['violated_constraints']} != {actual_violated}")
    
    is_valid = len(issues) == 0
    
    if logger and not is_valid:
        logger.warning(f"Result validation failed: {', '.join(issues)}")
    
    return is_valid, issues


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def format_evidence_summary(
    constraint: str,
    birth_position: int,
    birth_evidence: str,
    violation_position: Optional[int] = None,
    violation_evidence: Optional[str] = None
) -> str:
    """
    Format constraint evidence as human-readable summary.
    
    Args:
        constraint: Constraint text
        birth_position: Birth position
        birth_evidence: Birth evidence text
        violation_position: Violation position (optional)
        violation_evidence: Violation evidence text (optional)
    
    Returns:
        Formatted summary string
    """
    summary = f"Constraint: {constraint}\n"
    summary += f"Birth: Position {birth_position}\n"
    summary += f"  Evidence: {birth_evidence[:150]}...\n"
    
    if violation_position is not None and violation_evidence:
        summary += f"Violation: Position {violation_position}\n"
        summary += f"  Evidence: {violation_evidence[:150]}...\n"
    
    return summary


def compute_evaluation_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across multiple evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
    
    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {
            'total_statements': 0,
            'total_constraints': 0,
            'total_violations': 0,
            'avg_constraints_per_statement': 0.0,
            'avg_violations_per_statement': 0.0,
            'violation_rate': 0.0,
            'avg_execution_time': 0.0
        }
    
    total_statements = len(results)
    total_constraints = sum(r.get('total_constraints', 0) for r in results)
    total_violations = sum(r.get('violated_constraints', 0) for r in results)
    total_execution_time = sum(r.get('execution_time', 0.0) for r in results)
    
    return {
        'total_statements': total_statements,
        'total_constraints': total_constraints,
        'total_violations': total_violations,
        'avg_constraints_per_statement': total_constraints / total_statements,
        'avg_violations_per_statement': total_violations / total_statements,
        'violation_rate': total_violations / total_constraints if total_constraints > 0 else 0.0,
        'avg_execution_time': total_execution_time / total_statements
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Set up logging
    logger = setup_evaluation_logging()
    
    # Example: Parse JSON array
    json_output = '["Alice is tall", "Bob is short"]'
    constraints = parse_json_array_safe(json_output, logger)
    print(f"Parsed constraints: {constraints}")
    
    # Example: Validate constraints
    is_valid, issues = validate_constraint_list(constraints, logger)
    print(f"Valid: {is_valid}, Issues: {issues}")
    
    # Example: Parse binary keyword
    binary_output = "ESTABLISHES"
    result = parse_binary_keyword_safe(binary_output, "ESTABLISHES", "DOES_NOT_ESTABLISH", logger)
    print(f"Parsed binary: {result}")
    
    # Example: Create constraint record
    record = create_constraint_record(
        constraint="Alice is tall",
        birth_position=5,
        birth_evidence="Alice was very tall, over 6 feet.",
        violated=True,
        violation_position=10,
        violation_evidence="Alice was actually quite short.",
        is_revision=False
    )
    print(f"Constraint record: {record}")
    
    # Example: Log constraint birth
    log_constraint_birth(logger, "Alice is tall", 5, "Alice was very tall, over 6 feet.")
    
    # Example: Log violation
    log_violation(logger, "Alice is tall", 10, "Alice was actually quite short.", is_revision=False)
    
    # Example: Create evaluation result
    eval_result = create_evaluation_result(
        statement="Alice is tall and Bob is short.",
        prediction=1,
        constraints=[record],
        execution_time=15.3,
        metadata={'novel': 'test_novel'}
    )
    
    # Example: Log summary
    log_evaluation_summary(logger, 1, 1, 1, 15.3)
    
    # Example: Save result
    save_evaluation_result(eval_result, "outputs/test_result.json", logger)
    
    # Example: Validate result
    is_valid, issues = validate_evaluation_result(eval_result, logger)
    print(f"Result valid: {is_valid}, Issues: {issues}")
    
    print("\n✓ evaluation.py examples completed successfully!")
