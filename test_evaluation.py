"""
Test suite for evaluation.py

Tests all parsing, logging, and validation functions.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, '/Users/mohit/IIT_KDSH')

from evaluation import (
    setup_evaluation_logging,
    log_constraint_birth,
    log_violation,
    log_parse_error,
    log_evaluation_summary,
    parse_json_array_safe,
    parse_binary_keyword_safe,
    parse_plain_text_safe,
    create_constraint_record,
    create_evaluation_result,
    save_evaluation_result,
    validate_constraint_list,
    validate_position,
    validate_evaluation_result,
    format_evidence_summary,
    compute_evaluation_statistics
)


def test_parse_json_array():
    """Test JSON array parsing with various inputs."""
    print("\n" + "="*60)
    print("TEST: JSON Array Parsing")
    print("="*60)
    
    # Test 1: Valid JSON
    result1 = parse_json_array_safe('["Alice is tall", "Bob is short"]')
    assert result1 == ["Alice is tall", "Bob is short"], "Valid JSON failed"
    print("✓ Valid JSON array")
    
    # Test 2: JSON with markdown
    result2 = parse_json_array_safe('```json\n["Alice", "Bob"]\n```')
    assert result2 == ["Alice", "Bob"], "Markdown JSON failed"
    print("✓ JSON with markdown")
    
    # Test 3: Empty input
    result3 = parse_json_array_safe("")
    assert result3 == [], "Empty input should return []"
    print("✓ Empty input returns []")
    
    # Test 4: Malformed JSON
    result4 = parse_json_array_safe("Not a JSON array")
    assert result4 == [], "Malformed JSON should return []"
    print("✓ Malformed JSON returns []")
    
    # Test 5: JSON with empty strings (should be filtered)
    result5 = parse_json_array_safe('["Alice", "", "Bob"]')
    assert result5 == ["Alice", "Bob"], "Empty strings should be filtered"
    print("✓ Empty strings filtered")
    
    print("\n✓ JSON array parsing tests passed")


def test_parse_binary_keyword():
    """Test binary keyword parsing."""
    print("\n" + "="*60)
    print("TEST: Binary Keyword Parsing")
    print("="*60)
    
    # Test 1: Positive keyword
    result1 = parse_binary_keyword_safe("ESTABLISHES", "ESTABLISHES", "DOES_NOT_ESTABLISH")
    assert result1 is True, "Positive keyword should return True"
    print("✓ Positive keyword → True")
    
    # Test 2: Negative keyword
    result2 = parse_binary_keyword_safe("DOES_NOT_ESTABLISH", "ESTABLISHES", "DOES_NOT_ESTABLISH")
    assert result2 is False, "Negative keyword should return False"
    print("✓ Negative keyword → False")
    
    # Test 3: Ambiguous (both keywords)
    result3 = parse_binary_keyword_safe("ESTABLISHES but also DOES_NOT_ESTABLISH", "ESTABLISHES", "DOES_NOT_ESTABLISH")
    assert result3 is None, "Ambiguous should return None"
    print("✓ Ambiguous → None")
    
    # Test 4: Neither keyword
    result4 = parse_binary_keyword_safe("Maybe something", "ESTABLISHES", "DOES_NOT_ESTABLISH")
    assert result4 is None, "Neither keyword should return None"
    print("✓ Neither keyword → None")
    
    # Test 5: Case insensitive
    result5 = parse_binary_keyword_safe("establishes", "ESTABLISHES", "DOES_NOT_ESTABLISH")
    assert result5 is True, "Should be case insensitive"
    print("✓ Case insensitive")
    
    print("\n✓ Binary keyword parsing tests passed")


def test_parse_plain_text():
    """Test plain text parsing."""
    print("\n" + "="*60)
    print("TEST: Plain Text Parsing")
    print("="*60)
    
    # Test 1: Simple text
    result1 = parse_plain_text_safe("Alice tall OR Bob short")
    assert result1 == "Alice tall OR Bob short", "Plain text failed"
    print("✓ Simple text")
    
    # Test 2: Text with markdown
    result2 = parse_plain_text_safe("```\nAlice tall\n```")
    assert result2 == "Alice tall", "Markdown removal failed"
    print("✓ Markdown removed")
    
    # Test 3: Excessive newlines
    result3 = parse_plain_text_safe("Line 1\n\n\n\nLine 2")
    assert result3 == "Line 1\n\nLine 2", "Newline collapsing failed"
    print("✓ Excessive newlines collapsed")
    
    # Test 4: Empty input
    result4 = parse_plain_text_safe("")
    assert result4 == "", "Empty input should return empty string"
    print("✓ Empty input")
    
    print("\n✓ Plain text parsing tests passed")


def test_constraint_record():
    """Test constraint record creation."""
    print("\n" + "="*60)
    print("TEST: Constraint Record Creation")
    print("="*60)
    
    # Test 1: Basic record
    record1 = create_constraint_record(
        constraint="Alice is tall",
        birth_position=5,
        birth_evidence="Alice was very tall.",
        violated=False
    )
    assert record1['constraint'] == "Alice is tall", "Constraint mismatch"
    assert record1['birth_position'] == 5, "Position mismatch"
    assert record1['violated'] is False, "Violated flag mismatch"
    print("✓ Basic constraint record")
    
    # Test 2: Violated record
    record2 = create_constraint_record(
        constraint="Bob is short",
        birth_position=3,
        birth_evidence="Bob was short.",
        violated=True,
        violation_position=10,
        violation_evidence="Bob was actually tall.",
        is_revision=False
    )
    assert record2['violated'] is True, "Violated flag wrong"
    assert record2['violation_position'] == 10, "Violation position wrong"
    assert record2['is_revision'] is False, "Revision flag wrong"
    print("✓ Violated constraint record")
    
    print("\n✓ Constraint record tests passed")


def test_evaluation_result():
    """Test evaluation result creation and validation."""
    print("\n" + "="*60)
    print("TEST: Evaluation Result")
    print("="*60)
    
    # Create sample constraint
    constraint = create_constraint_record(
        constraint="Test constraint",
        birth_position=0,
        birth_evidence="Test evidence",
        violated=True,
        violation_position=5,
        violation_evidence="Violation evidence"
    )
    
    # Create result
    result = create_evaluation_result(
        statement="Test statement",
        prediction=1,
        constraints=[constraint],
        execution_time=10.5,
        metadata={'novel': 'test_novel'}
    )
    
    assert result['prediction'] == 1, "Prediction wrong"
    assert result['total_constraints'] == 1, "Total constraints wrong"
    assert result['violated_constraints'] == 1, "Violated count wrong"
    print("✓ Evaluation result created")
    
    # Validate result
    is_valid, issues = validate_evaluation_result(result)
    assert is_valid, f"Result should be valid, issues: {issues}"
    print("✓ Result validation passed")
    
    print("\n✓ Evaluation result tests passed")


def test_save_and_load_result():
    """Test saving and loading evaluation results."""
    print("\n" + "="*60)
    print("TEST: Save/Load Result")
    print("="*60)
    
    # Create temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_result.json"
        
        # Create result
        constraint = create_constraint_record(
            constraint="Test",
            birth_position=0,
            birth_evidence="Evidence"
        )
        result = create_evaluation_result(
            statement="Test",
            prediction=0,
            constraints=[constraint],
            execution_time=5.0
        )
        
        # Save
        save_evaluation_result(result, str(output_path))
        assert output_path.exists(), "Result file not created"
        print("✓ Result saved")
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['prediction'] == 0, "Loaded prediction wrong"
        assert loaded['statement'] == "Test", "Loaded statement wrong"
        print("✓ Result loaded correctly")
    
    print("\n✓ Save/load tests passed")


def test_validate_constraints():
    """Test constraint list validation."""
    print("\n" + "="*60)
    print("TEST: Constraint Validation")
    print("="*60)
    
    # Test 1: Valid constraints
    valid_constraints = ["Alice is tall", "Bob is short"]
    is_valid, issues = validate_constraint_list(valid_constraints)
    assert is_valid, f"Should be valid, issues: {issues}"
    print("✓ Valid constraints")
    
    # Test 2: Empty list
    is_valid, issues = validate_constraint_list([])
    assert not is_valid, "Empty list should be invalid"
    assert "Empty constraint list" in issues[0], "Wrong error message"
    print("✓ Empty list detected")
    
    # Test 3: Too short constraint
    short_constraints = ["Hi", "This is okay"]
    is_valid, issues = validate_constraint_list(short_constraints)
    assert not is_valid, "Too-short constraint should be invalid"
    print("✓ Too-short constraint detected")
    
    # Test 4: Duplicates
    duplicate_constraints = ["Alice is tall", "Alice is tall"]
    is_valid, issues = validate_constraint_list(duplicate_constraints)
    assert not is_valid, "Duplicates should be invalid"
    assert "duplicate" in issues[0].lower(), "Wrong error message"
    print("✓ Duplicates detected")
    
    print("\n✓ Constraint validation tests passed")


def test_validate_position():
    """Test position validation."""
    print("\n" + "="*60)
    print("TEST: Position Validation")
    print("="*60)
    
    # Test 1: Valid position
    assert validate_position(5, 100), "Valid position should pass"
    print("✓ Valid position")
    
    # Test 2: Negative position
    assert not validate_position(-1, 100), "Negative position should fail"
    print("✓ Negative position detected")
    
    # Test 3: Out of bounds
    assert not validate_position(100, 100), "Out of bounds should fail"
    print("✓ Out of bounds detected")
    
    # Test 4: Non-integer
    assert not validate_position(5.5, 100), "Non-integer should fail"
    print("✓ Non-integer detected")
    
    print("\n✓ Position validation tests passed")


def test_evidence_summary():
    """Test evidence summary formatting."""
    print("\n" + "="*60)
    print("TEST: Evidence Summary")
    print("="*60)
    
    summary = format_evidence_summary(
        constraint="Alice is tall",
        birth_position=5,
        birth_evidence="Alice was very tall, over 6 feet.",
        violation_position=10,
        violation_evidence="Alice was actually quite short."
    )
    
    assert "Alice is tall" in summary, "Constraint not in summary"
    assert "Position 5" in summary, "Birth position not in summary"
    assert "Position 10" in summary, "Violation position not in summary"
    print("✓ Evidence summary formatted")
    
    print("\n✓ Evidence summary tests passed")


def test_compute_statistics():
    """Test statistics computation."""
    print("\n" + "="*60)
    print("TEST: Statistics Computation")
    print("="*60)
    
    # Create sample results
    results = [
        {
            'total_constraints': 2,
            'violated_constraints': 1,
            'execution_time': 10.0
        },
        {
            'total_constraints': 3,
            'violated_constraints': 0,
            'execution_time': 15.0
        }
    ]
    
    stats = compute_evaluation_statistics(results)
    
    assert stats['total_statements'] == 2, "Statement count wrong"
    assert stats['total_constraints'] == 5, "Constraint count wrong"
    assert stats['total_violations'] == 1, "Violation count wrong"
    assert stats['avg_constraints_per_statement'] == 2.5, "Avg constraints wrong"
    assert stats['avg_execution_time'] == 12.5, "Avg time wrong"
    print("✓ Statistics computed correctly")
    
    # Test empty results
    empty_stats = compute_evaluation_statistics([])
    assert empty_stats['total_statements'] == 0, "Empty stats wrong"
    print("✓ Empty results handled")
    
    print("\n✓ Statistics tests passed")


def test_logging_setup():
    """Test logging setup."""
    print("\n" + "="*60)
    print("TEST: Logging Setup")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_evaluation_logging(log_dir=tmpdir)
        
        # Test logging
        log_constraint_birth(logger, "Test constraint", 5, "Test evidence")
        log_violation(logger, "Test constraint", 10, "Violation evidence", is_revision=False)
        log_parse_error(logger, "TEST_PROMPT", "bad output", "JSON", "Parse failed")
        log_evaluation_summary(logger, 2, 1, 1, 15.5)
        
        # Check log file created
        log_files = list(Path(tmpdir).glob("evaluation_*.log"))
        assert len(log_files) > 0, "Log file not created"
        print("✓ Log file created")
        
        # Check log contents
        with open(log_files[0], 'r') as f:
            log_content = f.read()
        
        assert "CONSTRAINT_BIRTH" in log_content, "Birth log not found"
        assert "VIOLATION" in log_content, "Violation log not found"
        assert "PARSE_ERROR" in log_content, "Parse error log not found"
        assert "EVALUATION_SUMMARY" in log_content, "Summary log not found"
        print("✓ Log contents correct")
    
    print("\n✓ Logging tests passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING evaluation.py - ALL FUNCTIONS")
    print("="*80)
    
    try:
        test_parse_json_array()
        test_parse_binary_keyword()
        test_parse_plain_text()
        test_constraint_record()
        test_evaluation_result()
        test_save_and_load_result()
        test_validate_constraints()
        test_validate_position()
        test_evidence_summary()
        test_compute_statistics()
        test_logging_setup()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED (11/11)")
        print("="*80)
        print("\nVerified Functionality:")
        print("  ✓ JSON array parsing (defensive)")
        print("  ✓ Binary keyword parsing (deterministic)")
        print("  ✓ Plain text parsing (clean)")
        print("  ✓ Constraint record creation")
        print("  ✓ Evaluation result structure")
        print("  ✓ Save/load results (JSON)")
        print("  ✓ Constraint validation")
        print("  ✓ Position validation")
        print("  ✓ Evidence summary formatting")
        print("  ✓ Statistics computation")
        print("  ✓ Structured logging (birth, violations, errors)")
        print("\n✓ evaluation.py ready for production!")
        
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
