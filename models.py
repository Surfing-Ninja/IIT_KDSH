"""
Model loading utilities for Qwen LLM, embedding model, and reranker.
Production-grade implementation with strict configuration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder

import config


def load_llm():
    """
    Load Qwen2.5-14B-Instruct with 4-bit NF4 quantization.
    NVIDIA T4 compatible. CUDA required.
    
    Returns:
        tuple: (model, tokenizer)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required. No GPU detected.")
    
    print(f"Loading {config.QWEN_MODEL_NAME} with 4-bit NF4 quantization...")
    
    # 4-bit NF4 quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.QWEN_MODEL_NAME,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.QWEN_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model.eval()
    
    memory_gb = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Qwen loaded on CUDA | Memory: {memory_gb:.2f} GB")
    
    return model, tokenizer


def load_embedder():
    """
    Load BAAI/bge-m3 embedding model.
    
    Returns:
        SentenceTransformer: bge-m3 model
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required. No GPU detected.")
    
    print(f"Loading {config.EMBEDDING_MODEL_NAME}...")
    
    embedder = SentenceTransformer(
        config.EMBEDDING_MODEL_NAME,
        device="cuda"
    )
    
    print(f"✓ Embedder loaded on CUDA | Dimension: {config.EMBEDDING_DIM}")
    
    return embedder


def load_reranker():
    """
    Load BAAI/bge-reranker-large for reranking.
    Must run BEFORE LLM evaluation.
    
    Returns:
        CrossEncoder: bge-reranker-large model
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required. No GPU detected.")
    
    print(f"Loading {config.RERANKER_MODEL_NAME}...")
    
    reranker = CrossEncoder(
        config.RERANKER_MODEL_NAME,
        max_length=512,
        device="cuda"
    )
    
    print(f"✓ Reranker loaded on CUDA")
    
    return reranker


def generate_with_qwen(model, tokenizer, prompt, max_new_tokens=None, debug=False):
    """
    Generate text with Qwen using deterministic settings.
    Temperature = 0.01 for near-deterministic output.
    
    Args:
        model: Qwen model
        tokenizer: Qwen tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        debug: Print debug information
        
    Returns:
        str: Generated response
    """
    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS
    
    # Format for chat model
    messages = [
        {"role": "system", "content": "You are a precise logical reasoning assistant. Always respond in the exact format requested."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    if debug:
        print("\n" + "="*80)
        print("DEBUG: PROMPT SENT TO MODEL:")
        print(text[:800])  # First 800 chars
        print("..." if len(text) > 800 else "")
        print("="*80)
    
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=config.TEMPERATURE,
            do_sample=config.DO_SAMPLE,
            top_p=config.TOP_P,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only new tokens
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if debug:
        print("\n" + "="*80)
        print("DEBUG: RAW MODEL OUTPUT:")
        print(response)
        print("="*80)
    
    return response.strip()


def load_nli_model():
    """
    Load RoBERTa-large-MNLI for contradiction detection (high-recall filter).
    
    This model is used BEFORE Qwen to filter out chunks that are clearly
    not contradictions. It has high recall (catches most contradictions)
    but lower precision (may flag non-contradictions).
    
    Purpose:
    - Acts as a fast, high-recall filter
    - Reduces load on Qwen by eliminating obvious non-contradictions
    - Trained specifically for entailment/contradiction tasks
    
    Returns:
        AutoModelForSequenceClassification: NLI model for contradiction detection
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = "roberta-large-mnli"
    
    print(f"Loading {model_name} for NLI contradiction filtering...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    model.eval()
    
    print(f"✓ NLI model loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    return model, tokenizer


def check_contradiction_nli(nli_model, nli_tokenizer, premise, hypothesis, return_score=False):
    """
    Use NLI model to check if hypothesis contradicts premise.
    
    This is a HIGH-RECALL filter - it will flag anything that looks
    oppositional, even if it's not a true contradiction.
    
    Args:
        nli_model: RoBERTa-large-MNLI model
        nli_tokenizer: Corresponding tokenizer
        premise: The constraint/established fact
        hypothesis: The chunk to check
        return_score: If True, return (is_contradiction, score); else just bool
        
    Returns:
        bool or tuple: True if potential contradiction, or (bool, float) if return_score=True
    """
    import config
    
    # Tokenize input
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move to same device as model
    inputs = {k: v.to(nli_model.device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = nli_model(**inputs)
        logits = outputs.logits
    
    # RoBERTa-large-MNLI labels: 0=contradiction, 1=neutral, 2=entailment
    # We use softmax to get probability distribution
    probs = torch.softmax(logits, dim=1)[0]
    contradiction_score = probs[0].item()  # Probability of contradiction
    
    # Return score or threshold decision
    if return_score:
        return contradiction_score >= config.NLI_WEAK_THRESHOLD, contradiction_score
    else:
        # High-recall mode: flag if score >= weak threshold
        return contradiction_score >= config.NLI_WEAK_THRESHOLD
