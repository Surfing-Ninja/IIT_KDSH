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
