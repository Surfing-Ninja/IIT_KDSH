"""
All LLM prompts for the constraint consistency pipeline.

STRICT REQUIREMENTS:
- All prompts produce deterministic outputs (temperature=0.0)
- All outputs are machine-parsable (JSON or binary)
- NO chain-of-thought reasoning
- NO explanations in output
- NO embedded logic in prompts
- Minimal, reusable prompt templates

Qwen2.5-14B-Instruct is used for:
1. Constraint extraction (JSON output)
2. Binary classification (ESTABLISHES/VIOLATES/etc.)
3. Query generation (plain text)
"""

# ============================================================================
# 1. CONSTRAINT EXTRACTION (JSON Output)
# ============================================================================

CONSTRAINT_EXTRACTION_PROMPT = """Extract logical constraints from the statement as a JSON list with type categorization.

IMPORTANT: Output ONLY the JSON array. Do NOT include explanations, reasoning, or any other text.

CONSTRAINT TYPES:
- belief: Character beliefs, opinions, or mental states
- prohibition: Rules, restrictions, or things that cannot happen
- motivation: Character goals, desires, or driving forces
- background_fact: Objective facts about setting, history, or entities
- fear: Character fears, anxieties, or things they avoid

RULES:
- Extract ONLY persistent facts (not events or opinions)
- Each constraint is atomic (one fact only)
- Each constraint is testable (can be verified in text)
- Assign appropriate type from the 5 categories above
- Output valid JSON array ONLY, no explanation, no reasoning

EXAMPLES:

Input: "John is a doctor who lives in Paris."
Output: [
  {{"type": "background_fact", "constraint": "John is a doctor"}},
  {{"type": "background_fact", "constraint": "John lives in Paris"}}
]

Input: "Sarah fears dogs and seeks revenge for her father's death."
Output: [
  {{"type": "fear", "constraint": "Sarah fears dogs"}},
  {{"type": "motivation", "constraint": "Sarah seeks revenge for her father's death"}}
]

Input: "The prisoner cannot leave the island and believes escape is impossible."
Output: [
  {{"type": "prohibition", "constraint": "The prisoner cannot leave the island"}},
  {{"type": "belief", "constraint": "The prisoner believes escape is impossible"}}
]

Now extract constraints from this statement. Output ONLY the JSON array:

Statement: {statement}

JSON:"""

# ============================================================================
# 2. ESTABLISHMENT CHECK (Binary Output)
# ============================================================================

ESTABLISHMENT_CHECK_PROMPT = """Does this passage establish or strongly reference the constraint?

Constraint: {constraint}

Passage:
{passage}

RULES:
- ESTABLISHES: The passage directly states, introduces, or clearly implies this constraint
- Accept strong mentions or clear references
- Only reject if constraint is absent or contradicted

EXAMPLES:

Constraint: "John is a doctor"
Passage: "Dr. John examined the patient in the hospital."
Answer: ESTABLISHES

Constraint: "learned to track animals"
Passage: "He spent years tracking animals across the plains."
Answer: ESTABLISHES

Constraint: "Sarah lives in Paris"
Passage: "Sarah walked through the streets of Paris near her apartment."
Answer: ESTABLISHES

Constraint: "The box is red"
Passage: "The blue box sat on the table."
Answer: DOES_NOT_ESTABLISH

Output ONLY one word (ESTABLISHES or DOES_NOT_ESTABLISH):"""

# ============================================================================
# 3. VIOLATION QUERY GENERATION (Plain Text Output)
# ============================================================================

VIOLATION_QUERY_PROMPT = """Generate a search query to find text that VIOLATES this constraint.

Constraint: {constraint}

RULES:
- Query should retrieve passages contradicting the constraint
- Use keywords for semantic/lexical search
- 5-15 words, focus on contradictory terms

EXAMPLES:

Constraint: "John is a doctor"
Query: John lawyer attorney profession career not doctor

Constraint: "The diamond is blue"
Query: diamond color red green yellow white not blue

Constraint: "Sarah lives in Paris"
Query: Sarah home residence city London Berlin not Paris

Now generate query for the given constraint (output ONLY the query text):"""

# ============================================================================
# 4. VIOLATION CHECK (Binary Output)
# ============================================================================

VIOLATION_CHECK_PROMPT = """Does this passage VIOLATE the constraint?

Constraint: {constraint}

Establishment context: {establishment_context}

Passage (later in narrative):
{passage}

RULES:
- VIOLATES: Passage directly contradicts the constraint (logical inconsistency)
- Does NOT qualify as violation if: synonyms, stylistic variation, or justified narrative change

EXAMPLES:

Constraint: "John is a doctor"
Establishment: "Dr. John treated patients at the hospital."
Passage: "John the lawyer argued the case in court."
Answer: VIOLATES

Constraint: "The box is red"
Establishment: "The red box was placed on the shelf."
Passage: "She opened the crimson container."
Answer: DOES_NOT_VIOLATE (synonym)

Constraint: "Sarah hates dogs"
Establishment: "Sarah always avoided dogs."
Passage: "After therapy, Sarah adopted a puppy."
Answer: DOES_NOT_VIOLATE (justified change)

Output ONLY one word (VIOLATES or DOES_NOT_VIOLATE):"""

# ============================================================================
# 5. REVISION CHECK (Binary Output)
# ============================================================================

REVISION_CHECK_PROMPT = """Is this an intentional REVISION or an inconsistent VIOLATION?

Constraint: {constraint}
Establishment: {establishment_context}

Later passage: {violation_passage}

RULES:
- REVISION: Explicit narrative change with acknowledgment or explanation
- VIOLATION: Contradiction with no narrative justification

EXAMPLES:

Constraint: "John is a doctor"
Establishment: "Dr. John worked at the hospital."
Passage: "John quit medicine last year and became a lawyer."
Answer: REVISION (explicit change)

Constraint: "Sarah lives in Paris"
Establishment: "Sarah's Parisian apartment was cozy."
Passage: "Sarah returned to her home in London."
Answer: VIOLATION (no explanation of move)

Constraint: "The box is red"
Establishment: "The red box was on the shelf."
Passage: "The box turned blue after the magic spell."
Answer: REVISION (explained change)

Output ONLY one word (REVISION or VIOLATION):"""

# ============================================================================
# 6. RETRIEVAL QUALITY CHECK (Binary Output)
# ============================================================================

RETRIEVAL_QUALITY_PROMPT = """Are the retrieved passages useful for finding violations of the constraint?

Constraint: {constraint}
Query: {query}

Retrieved passages:
{passages}

RULES:
- GOOD: Passages are relevant and could contain violations
- POOR: Passages are off-topic or clearly cannot contain violations

Output ONLY one word (GOOD or POOR):"""


# ============================================================================
# 7. QUERY REFINEMENT (Plain Text Output)
# ============================================================================

QUERY_REFINEMENT_PROMPT = """The search query did not retrieve useful passages. Generate a better query.

Original constraint: {constraint}
Original query: {original_query}

Retrieved passages (not useful):
{passages}

Generate a BETTER search query that will find violations of this constraint.

RULES:
- Use different keywords or phrasings
- Focus on what would contradict the constraint
- Keep query concise (max 20 words)
- No explanations, just the query

Better query:"""


# ============================================================================
# 8. GLOBAL SANITY CHECK (Binary Output)
# ============================================================================

SANITY_CHECK_PROMPT = """Does the evidence support the constraint assessment?

Constraint: {constraint}
Assessment: {assessment}

Evidence:
{evidence}

RULES:
- VALID: Evidence clearly supports the assessment
- INVALID: Evidence does not support or contradicts the assessment

Output ONLY one word (VALID or INVALID):"""


# ============================================================================
# PROMPT METADATA (for validation)
# ============================================================================

PROMPTS_METADATA = {
    "CONSTRAINT_EXTRACTION_PROMPT": {
        "output_format": "JSON array of objects",
        "example_output": '[{"type": "background_fact", "constraint": "John is a doctor"}]',
        "temperature": 0.0,
        "max_tokens": 512
    },
    "ESTABLISHMENT_CHECK_PROMPT": {
        "output_format": "Binary: ESTABLISHES or DOES_NOT_ESTABLISH",
        "example_output": "ESTABLISHES",
        "temperature": 0.0,
        "max_tokens": 10
    },
    "VIOLATION_QUERY_PROMPT": {
        "output_format": "Plain text query",
        "example_output": "John lawyer attorney not doctor",
        "temperature": 0.0,
        "max_tokens": 100
    },
    "VIOLATION_CHECK_PROMPT": {
        "output_format": "Binary: VIOLATES or DOES_NOT_VIOLATE",
        "example_output": "VIOLATES",
        "temperature": 0.0,
        "max_tokens": 10
    },
    "REVISION_CHECK_PROMPT": {
        "output_format": "Binary: REVISION or VIOLATION",
        "example_output": "REVISION",
        "temperature": 0.0,
        "max_tokens": 10
    },
    "RETRIEVAL_QUALITY_PROMPT": {
        "output_format": "Binary: GOOD or POOR",
        "example_output": "GOOD",
        "temperature": 0.0,
        "max_tokens": 10
    },
    "QUERY_REFINEMENT_PROMPT": {
        "output_format": "Plain text query",
        "example_output": "John medical practice physician",
        "temperature": 0.0,
        "max_tokens": 50
    },
    "SANITY_CHECK_PROMPT": {
        "output_format": "Binary: VALID or INVALID",
        "example_output": "VALID",
        "temperature": 0.0,
        "max_tokens": 10
    }
}
