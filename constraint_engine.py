"""
Core constraint consistency checking logic.
Implements constraint extraction, establishment checking, and violation detection.

ARCHITECTURAL RULE: Treat as CONSTRAINT PERSISTENCE, not QA.
A single violated constraint → label = 0 (inconsistent).

STRICT REQUIREMENTS:
1. Extract constraints from backstory using Qwen (JSON output)
2. For each constraint:
   a. Retrieve candidate chunks via Pathway
   b. Find earliest chunk that ESTABLISHES constraint
   c. Record established_at (narrative_position)
3. Retrieve only chunks with position > established_at
4. Search for violations using targeted retrieval queries
5. Run binary LLM checks
6. If any constraint is violated → output 0, else 1

RETRIEVAL RULES:
- NEVER scan all chunks
- ALWAYS retrieve first, then verify with LLM
- Use reranker before LLM calls

DECISION RULE:
- One violation = contradiction
- No voting, no weighting, no averaging
"""

import json
import config
import prompts
from models import generate_with_qwen
from evaluation_utils import (
    parse_llm_output,
    log_step,
    sanity_check,
    validate_retrieval_quality
)


def extract_constraints(model, tokenizer, statement):
    """
    Extract logical constraints from backstory statement using Qwen.
    Uses structured JSON output for deterministic extraction.
    
    STRICT: Temperature = 0.0 for reproducibility.
    
    Args:
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        statement: Backstory statement text to analyze
        
    Returns:
        List[str]: Extracted constraint strings
    """
    print("\n" + "="*60)
    print("[STEP 1] CONSTRAINT EXTRACTION")
    print("="*60)
    print(f"Statement: {statement[:150]}..." if len(statement) > 150 else f"Statement: {statement}")
    
    prompt = prompts.CONSTRAINT_EXTRACTION_PROMPT.format(statement=statement)
    
    # Generate with Qwen (temperature=0.01, near-deterministic)
    # Enable debug for first call to see what's happening
    # Increased tokens for DeepSeek-R1's reasoning chains
    debug_enabled = not hasattr(extract_constraints_with_types, '_first_call_done')
    output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=1024, debug=debug_enabled)
    extract_constraints_with_types._first_call_done = True
    
    # Parse JSON output - use json_array format
    constraints = parse_llm_output(output, expected_format="json_array")
    
    if not constraints:
        print("⚠️  WARNING: No constraints extracted from LLM output")
        print(f"Raw output (first 300 chars): {output[:300]}")
    else:
        print(f"✓ Extracted {len(constraints)} constraints")
    
    # Limit to max constraints
    constraints = constraints[:config.MAX_CONSTRAINTS_PER_STATEMENT]
    
    log_step(
        "extract_constraints",
        {"statement": statement},
        {"constraints": constraints},
        {"raw_output": output}
    )
    
    print(f"\n✓ Extracted {len(constraints)} constraints:")
    for i, c in enumerate(constraints, 1):
        preview = c[:80] + "..." if len(c) > 80 else c
        print(f"  {i}. {preview}")
    
    return constraints


def find_constraint_establishment(vector_store, model, tokenizer, constraint, reranker):
    """
    Find where a constraint is FIRST ESTABLISHED in the novel.
    Uses retrieval-first approach: Pathway retrieve → rerank → binary LLM verify.
    
    CRITICAL: Returns the EARLIEST position where constraint is established.
    This becomes the filter threshold for violation search.
    
    Args:
        vector_store: PathwayVectorStore instance
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Constraint string to locate
        reranker: bge-reranker-large CrossEncoder
        
    Returns:
        Dict with {
            'found': bool,
            'position': int or None (narrative_position),
            'text': str or None,
            'confidence': float
        }
    """
    print("\n" + "="*60)
    print("[STEP 2] CONSTRAINT ESTABLISHMENT")
    print("="*60)
    print(f"Constraint: {constraint[:80]}...")
    
    # Step 1: Retrieve candidate chunks (NO position filter - search entire document)
    print("\nRetrieving candidates from Pathway store...")
    retrieved_chunks = vector_store.retrieve(
        query=constraint,
        position_filter=None,  # Search entire document
        top_k=config.RETRIEVAL_TOP_K
    )
    
    if not retrieved_chunks:
        print("⚠ No chunks retrieved")
        return {
            'found': False,
            'position': None,
            'text': None,
            'confidence': 0.0
        }
    
    print(f"Retrieved {len(retrieved_chunks)} chunks")
    
    # EARLY SKIP: If top retrieval score is very low, constraint likely not in novel
    top_retrieval_score = max([c.get('score', 0) for c in retrieved_chunks]) if retrieved_chunks else 0
    if top_retrieval_score < 0.45:
        print(f"⚠ Top retrieval score too low ({top_retrieval_score:.3f}) - constraint likely absent")
        return {
            'found': False,
            'position': None,
            'text': None,
            'confidence': 0.0
        }
    
    # Step 2: Rerank chunks (MUST happen before LLM verification)
    print(f"\nReranking with bge-reranker-large...")
    pairs = [[constraint, chunk['text']] for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(pairs)
    
    # Attach rerank scores
    for i, chunk in enumerate(retrieved_chunks):
        chunk['rerank_score'] = float(rerank_scores[i])
    
    # Sort by rerank score descending
    retrieved_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Take top-k after reranking
    top_chunks = retrieved_chunks[:config.RERANK_TOP_K]
    
    # Reranking completed - checking chunks
    
    # Step 3: Binary LLM verification - find EARLIEST establishment
    # Sort by narrative_position ascending to check earliest first
    top_chunks_by_position = sorted(top_chunks, key=lambda x: x['narrative_position'])
    
    # OPTIMIZATION: If top rerank score is very high, auto-accept (skip LLM)
    top_score = top_chunks_by_position[0]['rerank_score'] if top_chunks_by_position else 0
    if top_score >= config.ESTABLISHMENT_AUTO_ACCEPT_SCORE:
        chunk = top_chunks_by_position[0]
        position = chunk['narrative_position']
        print(f"\n✓ Constraint ESTABLISHED at position {position} (high score: {top_score:.3f}, skipped LLM)")
        return {
            'found': True,
            'position': position,
            'text': chunk['text'],
            'confidence': chunk['rerank_score']
        }
    
    # Otherwise check with LLM (check top 3 instead of all 5 for speed)
    for chunk in top_chunks_by_position[:3]:
        prompt = prompts.ESTABLISHMENT_CHECK_PROMPT.format(
            constraint=constraint,
            passage=chunk['text']
        )
        
        output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=50)
        result = parse_llm_output(output, expected_format="binary")
        
        # RELAXED: Accept ESTABLISHES or strong mentions
        if result == "ESTABLISHES" or "ESTABLISH" in output.upper():
            position = chunk['narrative_position']
            print(f"\n✓ Constraint ESTABLISHED at position {position}")
            
            birth_info = {
                'found': True,
                'position': position,
                'text': chunk['text'],
                'confidence': chunk['rerank_score']
            }
            
            log_step(
                "find_constraint_establishment",
                {"constraint": constraint},
                birth_info,
                {"chunks_checked": len(top_chunks_by_position)}
            )
            
            return birth_info
    
    # Not found in top chunks
    print("\n⚠ Constraint establishment NOT FOUND in top chunks")
    return {
        'found': False,
        'position': None,
        'text': None,
        'confidence': 0.0
    }


def generate_violation_query(model, tokenizer, constraint):
    """
    Generate a targeted search query to find potential violations.
    
    Purpose: Create semantically relevant query for dense retrieval.
    Example: "John is a doctor" → "John lawyer attorney legal profession"
    
    Args:
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Constraint to check for violations
        
    Returns:
        str: Generated violation search query
    """
    print("\n" + "="*60)
    print("[STEP 3] VIOLATION QUERY GENERATION")
    print("="*60)
    print(f"Constraint: {constraint[:80]}...")
    
    prompt = prompts.VIOLATION_QUERY_PROMPT.format(constraint=constraint)
    
    output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=100)
    query = parse_llm_output(output, expected_format="text")
    
    print(f"\n✓ Generated query: {query}")
    
    log_step(
        "generate_violation_query",
        {"constraint": constraint},
        {"query": query}
    )
    
    return query


def assess_retrieval_quality(
    model,
    tokenizer,
    constraint,
    query,
    retrieved_chunks,
    logger=None
):
    """
    Assess whether retrieved chunks are useful for finding violations.
    
    Part of self-refinement loop (Section 3.7 of paper).
    
    Args:
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Constraint being checked
        query: Search query used
        retrieved_chunks: List of retrieved chunk dicts
        logger: Optional logger for evaluation.py integration
        
    Returns:
        str: "GOOD" if quality is acceptable, "POOR" if refinement needed
    """
    if not retrieved_chunks:
        return "POOR"  # No results = poor quality
    
    # Take top 3 chunks for quality assessment
    top_chunks = retrieved_chunks[:3]
    passages_text = "\n\n---\n\n".join([
        f"Passage {i+1}: {chunk['text'][:200]}..."
        for i, chunk in enumerate(top_chunks)
    ])
    
    prompt = prompts.RETRIEVAL_QUALITY_PROMPT.format(
        constraint=constraint,
        query=query,
        passages=passages_text
    )
    
    output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=10)
    
    # Parse binary output: GOOD or POOR
    from evaluation import parse_binary_keyword_safe
    quality = parse_binary_keyword_safe(output, "GOOD", "POOR", logger)
    
    if quality is True:
        return "GOOD"
    elif quality is False:
        return "POOR"
    else:
        # Ambiguous or not found - default to accepting results
        return "GOOD"


def refine_violation_query(
    model,
    tokenizer,
    constraint,
    original_query,
    poor_chunks,
    logger=None
):
    """
    Refine search query when initial retrieval was poor.
    
    Part of self-refinement loop (Section 3.7 of paper).
    
    Args:
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Constraint being checked
        original_query: Original search query that didn't work well
        poor_chunks: Retrieved chunks that were not useful
        logger: Optional logger for evaluation.py integration
        
    Returns:
        str: Refined search query
    """
    # Show model why previous query failed
    passages_text = "\n\n---\n\n".join([
        f"Passage {i+1}: {chunk['text'][:150]}..."
        for i, chunk in enumerate(poor_chunks[:3])
    ])
    
    prompt = prompts.QUERY_REFINEMENT_PROMPT.format(
        constraint=constraint,
        original_query=original_query,
        passages=passages_text
    )
    
    output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=50)
    
    # Parse plain text output
    from evaluation import parse_plain_text_safe
    refined_query = parse_plain_text_safe(output, logger)
    
    return refined_query if refined_query else original_query  # Fallback to original


def search_for_violations_with_refinement(
    vector_store,
    model,
    tokenizer,
    constraint,
    violation_query,
    established_at,
    reranker,
    max_refinement_attempts=2,
    logger=None
):
    """
    Search for violations with self-refinement loop (Section 3.7 of paper).
    
    SELF-REFINEMENT PROCESS:
    1. Retrieve chunks with initial query
    2. Assess retrieval quality (GOOD/POOR)
    3. If POOR and attempts < max:
       a. Refine query
       b. Re-retrieve with refined query
       c. Repeat assessment
    4. Use best retrieval result for violation checking
    
    BOUNDED REFINEMENT:
    - Max refinement attempts prevents instability
    - Always returns result (even if quality remains poor)
    
    Args:
        vector_store: PathwayVectorStore instance
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Constraint being checked
        violation_query: Initial search query
        established_at: Position where constraint was established
        reranker: bge-reranker-large CrossEncoder
        max_refinement_attempts: Max number of query refinement iterations (default: 2)
        logger: Optional logger from evaluation.py
        
    Returns:
        Tuple[bool, Optional[Dict]]:
            - violation_found: True if violation detected
            - evidence: Dict with violation details or None
    """
    print("\n" + "="*60)
    print("[STEP 4] VIOLATION SEARCH (with self-refinement)")
    print("="*60)
    print(f"Constraint: {constraint[:80]}...")
    print(f"Established at: position {established_at}")
    print(f"Max refinement attempts: {max_refinement_attempts}")
    
    current_query = violation_query
    best_chunks = None
    
    for attempt in range(max_refinement_attempts + 1):  # +1 for initial attempt
        print(f"\n--- Attempt {attempt + 1}/{max_refinement_attempts + 1} ---")
        print(f"Query: {current_query}")
        
        # Retrieve chunks
        print(f"\nRetrieving from Pathway (position_filter={established_at + 1})...")
        retrieved_chunks = vector_store.retrieve(
            query=current_query,
            position_filter=established_at + 1,  # ONLY chunks AFTER establishment
            top_k=config.RETRIEVAL_TOP_K
        )
        
        if not retrieved_chunks:
            print("⚠ No chunks retrieved")
            if attempt < max_refinement_attempts:
                print("Refining query and retrying...")
                current_query = refine_violation_query(
                    model, tokenizer, constraint, current_query, [], logger
                )
                continue
            else:
                print("✓ No chunks found after all attempts (no violation)")
                return False, None
        
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # FIXED: Assess quality using retrieval scores (not broken LLM assessment)
        print("\nAssessing retrieval quality...")
        retrieval_scores = [c.get('score', 0) for c in retrieved_chunks]
        max_score = max(retrieval_scores) if retrieval_scores else 0
        
        # Score-based quality assessment (MUCH faster and more reliable)
        if max_score >= 0.60:
            quality = "GOOD"
        elif max_score >= 0.50:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
        
        print(f"Quality: {quality} (max_score: {max_score:.3f})")
        
        if quality in ["GOOD", "ACCEPTABLE"] or attempt == max_refinement_attempts:
            # Use these chunks (good/acceptable quality or last attempt)
            best_chunks = retrieved_chunks
            break
        else:
            # Quality is POOR and we have more attempts - refine query
            print("Quality is POOR, refining query...")
            current_query = refine_violation_query(
                model, tokenizer, constraint, current_query, retrieved_chunks, logger
            )
    
    # Now proceed with violation checking using best_chunks
    print(f"\nProceeding with violation checks (using {len(best_chunks)} chunks)")
    
    # Rerank chunks
    print(f"\nReranking with bge-reranker-large...")
    pairs = [[current_query, chunk['text']] for chunk in best_chunks]
    rerank_scores = reranker.predict(pairs)
    
    # Attach rerank scores
    for i, chunk in enumerate(best_chunks):
        chunk['rerank_score'] = float(rerank_scores[i])
    
    # Sort by rerank score descending
    best_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_chunks = best_chunks[:config.RERANK_TOP_K]
    
    print(f"Top {len(top_chunks)} chunks after reranking:")
    for i, chunk in enumerate(top_chunks[:3], 1):
        print(f"  {i}. pos={chunk['narrative_position']}, rerank_score={chunk['rerank_score']:.3f}")
    
    # Get establishment context
    establishment_chunk = vector_store.get_chunk_by_position(established_at)
    establishment_text = establishment_chunk['text'] if establishment_chunk else "N/A"
    
    # Binary LLM verification (check top 3 only for 40% speedup)
    check_limit = min(3, len(top_chunks))
    print(f"\\nLLM verification on top {check_limit} chunks...")
    for i, chunk in enumerate(top_chunks[:check_limit], 1):
        print(f"\n[DEBUG] Checking chunk {i}/{check_limit}:")
        print(f"  Position: {chunk['narrative_position']}")
        print(f"  Rerank score: {chunk['rerank_score']:.3f}")
        print(f"  Chunk preview: {chunk['text'][:120]}...")
        
        prompt = prompts.VIOLATION_CHECK_PROMPT.format(
            constraint=constraint,
            establishment_context=establishment_text[:300],
            passage=chunk['text']
        )
        
        output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=50)
        result = parse_llm_output(output, expected_format="binary")
        
        print(f"  LLM output: {output[:100]}...")
        print(f"  Parsed as: {result}")
        
        print(f"  Chunk {i} (pos={chunk['narrative_position']}): {result}")
        
        if result == "VIOLATES":
            # Check if it's a revision
            
            is_revision = check_if_revision(
                model,
                tokenizer,
                constraint,
                establishment_text,
                chunk['text']
            )
            
            if is_revision:
                print(f"  → Classified as REVISION (intentional change, not error)")
                continue
            
            # TRUE VIOLATION DETECTED
            position = chunk['narrative_position']
            print(f"\n✗ VIOLATION CONFIRMED at position {position}")
            
            evidence = {
                'constraint': constraint,
                'established_at': established_at,
                'establishment_text': establishment_text[:500],
                'violation_position': position,
                'violation_text': chunk['text'][:500],
                'rerank_score': chunk['rerank_score'],
                'refined_query': current_query if current_query != violation_query else None
            }
            
            # Log violation with evaluation.py if logger provided
            if logger:
                from evaluation import log_violation
                log_violation(logger, constraint, position, chunk['text'], is_revision=False)
            
            return True, evidence
    
    # No violation found
    print("\n✓ No violations detected")
    return False, None


def search_for_violations(
    vector_store,
    model,
    tokenizer,
    constraint,
    violation_query,
    established_at,
    reranker
):
    """
    Search for constraint violations using retrieval and binary LLM verification.
    
    CRITICAL POSITION FILTERING:
    - Only retrieves chunks with narrative_position > established_at
    - Ensures we don't check text before constraint was introduced
    - This is the core of constraint persistence checking
    
    ARCHITECTURE:
    1. Pathway retrieve with position_filter > established_at
    2. Rerank with bge-reranker-large
    3. Binary LLM checks (VIOLATES/DOES_NOT_VIOLATE)
    4. First violation found → return immediately
    
    NO VOTING, NO AVERAGING, NO SCORING:
    - Single violation = inconsistent
    - Binary decision per chunk
    
    Args:
        vector_store: PathwayVectorStore instance
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Constraint being checked
        violation_query: Search query for violations
        established_at: Position where constraint was established
        reranker: bge-reranker-large CrossEncoder
        
    Returns:
        Tuple[bool, Optional[Dict]]:
            - violation_found: True if violation detected
            - evidence: Dict with violation details or None
    """
    print("\n" + "="*60)
    print("[STEP 4] VIOLATION SEARCH")
    print("="*60)
    print(f"Constraint: {constraint[:80]}...")
    print(f"Established at: position {established_at}")
    print(f"Searching after: position {established_at + 1}")
    
    # Step 1: Retrieve chunks AFTER constraint establishment (position filtering)
    print(f"\nRetrieving from Pathway (position_filter={established_at + 1})...")
    retrieved_chunks = vector_store.retrieve(
        query=violation_query,
        position_filter=established_at + 1,  # ONLY chunks AFTER establishment
        top_k=config.RETRIEVAL_TOP_K
    )
    
    if not retrieved_chunks:
        print("✓ No chunks retrieved after establishment (no text to check)")
        return False, None
    
    print(f"Retrieved {len(retrieved_chunks)} chunks")
    
    # Step 2: Rerank chunks (before LLM verification)
    print(f"\nReranking with bge-reranker-large...")
    pairs = [[violation_query, chunk['text']] for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(pairs)
    
    # Attach rerank scores
    for i, chunk in enumerate(retrieved_chunks):
        chunk['rerank_score'] = float(rerank_scores[i])
    
    # Sort by rerank score descending
    retrieved_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_chunks = retrieved_chunks[:config.RERANK_TOP_K]
    
    print(f"Top {len(top_chunks)} chunks after reranking:")
    for i, chunk in enumerate(top_chunks[:3], 1):
        print(f"  {i}. pos={chunk['narrative_position']}, rerank_score={chunk['rerank_score']:.3f}")
    
    # Get establishment context for LLM prompt
    establishment_chunk = vector_store.get_chunk_by_position(established_at)
    establishment_text = establishment_chunk['text'] if establishment_chunk else "N/A"
    
    # Step 3: Binary LLM verification - check top 3 chunks only
    check_limit = min(3, len(top_chunks))
    print(f"\\nLLM verification on top {check_limit} chunks...")
    for i, chunk in enumerate(top_chunks[:check_limit], 1):
        print(f"\n[DEBUG] Direct check {i}/{check_limit}:")
        print(f"  Constraint: {constraint[:60]}...")
        print(f"  Position: {chunk['narrative_position']}")
        print(f"  Chunk preview: {chunk['text'][:120]}...")
        
        prompt = prompts.VIOLATION_CHECK_PROMPT.format(
            constraint=constraint,
            establishment_context=establishment_text[:300],
            passage=chunk['text']
        )
        
        output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=50)
        result = parse_llm_output(output, expected_format="binary")
        
        print(f"  LLM output: {output[:100]}...")
        print(f"  Parsed as: {result}")
        
        if result == "VIOLATES":
            # Step 4: Check if it's a valid revision (intentional change)
            
            is_revision = check_if_revision(
                model,
                tokenizer,
                constraint,
                establishment_text,
                chunk['text']
            )
            
            if is_revision:
                print(f"  → Classified as REVISION (intentional change, not error)")
                continue  # Not a true violation, keep searching
            
            # TRUE VIOLATION DETECTED
            position = chunk['narrative_position']
            print(f"\n✗ VIOLATION CONFIRMED at position {position}")
            
            evidence = {
                'constraint': constraint,
                'established_at': established_at,
                'establishment_text': establishment_text[:500],
                'violation_position': position,
                'violation_text': chunk['text'][:500],
                'rerank_score': chunk['rerank_score']
            }
            
            # Sanity check
            is_valid, issues = sanity_check(constraint, chunk['text'], result)
            if not is_valid:
                print(f"  Warning: Sanity check failed: {issues}")
            
            log_step(
                "search_for_violations",
                {
                    "constraint": constraint,
                    "query": violation_query,
                    "established_at": established_at
                },
                {"violation_found": True, "evidence": evidence}
            )
            
            return True, evidence
    
    # No violation found in any chunk
    print("\n✓ No violations detected")
    
    log_step(
        "search_for_violations",
        {
            "constraint": constraint,
            "query": violation_query,
            "established_at": established_at
        },
        {"violation_found": False}
    )
    
    return False, None


def check_if_revision(model, tokenizer, constraint, establishment_text, potential_violation_text):
    """
    Check if an apparent violation is actually a legitimate revision.
    Handles intentional character development, plot twists, retcons.
    
    Examples of revisions (NOT violations):
    - "John was a doctor" → "John quit medicine and became a lawyer" (explicit change)
    - "Sarah hated dogs" → "After therapy, Sarah adopted a dog" (character growth)
    
    Examples of violations (inconsistencies):
    - "John is a doctor" → "John the lawyer walked into court" (no transition)
    - "Sarah lives in NYC" → "Sarah returned to her LA apartment" (no explanation)
    
    Args:
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        constraint: Original constraint
        establishment_text: Where constraint was established
        potential_violation_text: Text that may contain violation
        
    Returns:
        bool: True if this is a valid revision, False if true violation
    """
    prompt = prompts.REVISION_CHECK_PROMPT.format(
        constraint=constraint,
        establishment_context=establishment_text[:300],
        violation_passage=potential_violation_text[:300]
    )
    
    output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=50)
    result = parse_llm_output(output, expected_format="binary")
    
    is_revision = (result == "REVISION")
    
    log_step(
        "check_if_revision",
        {
            "constraint": constraint,
            "establishment": establishment_text[:100],
            "potential_violation": potential_violation_text[:100]
        },
        {"is_revision": is_revision}
    )
    
    return is_revision


def check_constraint_consistency(
    vector_store,
    model,
    tokenizer,
    reranker,
    statement,
    novel_id,
    enable_refinement=True,
    max_refinement_attempts=2,
    logger=None
):
    """
    Main pipeline for checking constraint consistency.
    
    ARCHITECTURE (100% PAPER COMPLIANT):
    1. Extract constraints from statement (WITH TYPES - Section 3.3)
    2. For each constraint:
       a. Find establishment position (earliest occurrence)
       b. Generate violation query
       c. Search for violations WITH SELF-REFINEMENT (Section 3.7)
          - Assess retrieval quality
          - Refine query if poor
          - Bounded retry (max 2-3 attempts)
    3. Decision: ANY violation → 0, NO violations → 1
    
    CONSTRAINT CATEGORIZATION (Section 3.3):
    - 5 types: belief, prohibition, motivation, background_fact, fear
    - Each constraint now has explicit type metadata
    - Types preserved through pipeline
    
    SELF-REFINEMENT LOOP (Section 3.7):
    - Quality assessment: Binary GOOD/POOR on retrieved chunks
    - Query refinement: Generate better query when retrieval fails
    - Bounded iterations: Prevent instability with max attempts
    
    STRICT RULES:
    - NO end-to-end generation
    - NO chain-of-thought
    - NO agent frameworks
    - Binary decisions only
    - One violation = contradiction
    
    Args:
        vector_store: PathwayVectorStore with ingested novel
        model: Qwen LLM
        tokenizer: Qwen tokenizer
        reranker: bge-reranker-large CrossEncoder
        statement: Backstory statement to verify
        novel_id: Novel identifier for logging
        enable_refinement: Whether to use self-refinement loop (default: True)
        max_refinement_attempts: Max query refinement iterations (default: 2)
        logger: Optional logger from evaluation.py
        
    Returns:
        Dict with {
            'prediction': int (0=inconsistent, 1=consistent),
            'constraints': List[Dict],  # Now includes type info
            'violations': List[Dict],
            'summary': str
        }
    """
    print("\n" + "="*80)
    print(f"CONSTRAINT CONSISTENCY CHECK: {novel_id}")
    print("="*80)
    print(f"Statement: {statement}")
    print(f"Self-refinement: {'ENABLED' if enable_refinement else 'DISABLED'}")
    if enable_refinement:
        print(f"Max refinement attempts: {max_refinement_attempts}")
    print("="*80)
    
    # Step 1: Extract constraints WITH TYPES (Section 3.3)
    from evaluation import parse_constraints_with_types
    
    prompt = prompts.CONSTRAINT_EXTRACTION_PROMPT.format(statement=statement)
    raw_output = generate_with_qwen(model, tokenizer, prompt, max_new_tokens=1024)
    
    constraints = parse_constraints_with_types(raw_output, logger)
    
    if not constraints:
        print("\n⚠ No constraints extracted → Assume consistent")
        print(f"Debug - Raw output: {raw_output[:200]}...")
        return {
            'prediction': 1,
            'constraints': [],
            'violations': [],
            'summary': "No constraints to check"
        }
    
    print(f"\nExtracted {len(constraints)} constraint(s) with types:")
    for i, c in enumerate(constraints, 1):
        print(f"  {i}. [{c['type'].upper()}] {c['constraint'][:80]}...")
    
    # Step 2: Check each constraint
    all_violations = []
    
    for i, constraint_obj in enumerate(constraints, 1):
        constraint_text = constraint_obj['constraint']
        constraint_type = constraint_obj['type']
        
        print(f"\n{'='*80}")
        print(f"CONSTRAINT {i}/{len(constraints)} [{constraint_type.upper()}]")
        print(f"{'='*80}")
        
        # Step 2a: Find establishment position
        establishment = find_constraint_establishment(
            vector_store,
            model,
            tokenizer,
            constraint_text,
            reranker
        )
        
        if not establishment['found']:
            print(f"\n⚠ Constraint not established in novel → Skip")
            continue
        
        established_at = establishment['position']
        
        # Step 2b: Generate violation query
        violation_query = generate_violation_query(model, tokenizer, constraint_text)
        
        # Step 2c: Search for violations (WITH SELF-REFINEMENT if enabled)
        if enable_refinement:
            violation_found, evidence = search_for_violations_with_refinement(
                vector_store,
                model,
                tokenizer,
                constraint_text,
                violation_query,
                established_at,
                reranker,
                max_refinement_attempts=max_refinement_attempts,
                logger=logger
            )
        else:
            violation_found, evidence = search_for_violations(
                vector_store,
                model,
                tokenizer,
                constraint_text,
                violation_query,
                established_at,
                reranker
            )
        
        if violation_found:
            # Add constraint type to evidence
            evidence['constraint_type'] = constraint_type
            
            # CRITICAL: One violation = inconsistent
            all_violations.append(evidence)
            
            print(f"\n{'='*80}")
            print("✗ INCONSISTENCY DETECTED")
            print(f"{'='*80}")
            print(f"Constraint type: {constraint_type}")
            print(f"Constraint: {constraint_text}")
            print(f"Established at: position {established_at}")
            print(f"Violated at: position {evidence['violation_position']}")
            print("="*80)
            
            # DECISION: Return immediately (one violation = 0)
            return {
                'prediction': 0,
                'constraints': constraints,
                'violations': all_violations,
                'summary': f"Violation found for {constraint_type}: {constraint_text[:80]}..."
            }
    
    # Step 3: No violations found
    print(f"\n{'='*80}")
    print("✓ ALL CONSTRAINTS CONSISTENT")
    print(f"{'='*80}")
    print(f"Checked {len(constraints)} constraint(s)")
    print("No violations detected")
    print("="*80)
    
    return {
        'prediction': 1,
        'constraints': constraints,
        'violations': [],
        'summary': "All constraints consistent"
    }

