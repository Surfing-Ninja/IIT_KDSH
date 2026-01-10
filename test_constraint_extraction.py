"""
Test constraint extraction in isolation to debug issues.
Run this before the full pipeline.
"""

import torch
from models import load_llm, generate_with_qwen
from evaluation_utils import parse_llm_output
import prompts

def test_constraint_extraction():
    print("\n" + "="*80)
    print("TESTING CONSTRAINT EXTRACTION")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_llm()
    print("✓ Model loaded")
    
    # Test cases
    test_cases = [
        {
            "name": "Simple case",
            "statement": "Thalcave's mother died giving birth. He spent his childhood learning to track animals on the plains."
        },
        {
            "name": "Complex case",
            "statement": "Thalcave's people faded as colonists advanced; his father, last of the tribal guides, knew the pampas geography and animal ways, while his mother died giving birth. Boyhood was spent roaming the plains with his father, learning to track, tame horses and steer by the stars."
        },
        {
            "name": "Doctor example",
            "statement": "John is a doctor who lives in Paris."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print("\n" + "="*80)
        print(f"TEST CASE {i}: {test['name']}")
        print("="*80)
        print(f"Statement: {test['statement'][:100]}...")
        
        # Format prompt
        prompt = prompts.CONSTRAINT_EXTRACTION_PROMPT.format(statement=test['statement'])
        
        # Generate (with debug on first call)
        debug = (i == 1)
        output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=512, debug=debug)
        
        if not debug:
            print(f"\nRaw output: {output}")
        
        # Parse
        constraints = parse_llm_output(output, expected_format="json_array")
        
        # Results
        print(f"\n{'='*80}")
        if constraints:
            print(f"✅ SUCCESS: Extracted {len(constraints)} constraints")
            for j, c in enumerate(constraints, 1):
                print(f"  {j}. {c}")
        else:
            print(f"❌ FAILED: No constraints extracted")
        print("="*80)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_constraint_extraction()
