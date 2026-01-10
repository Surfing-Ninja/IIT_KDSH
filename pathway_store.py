"""
Pathway-based vector store with position-aware retrieval.
Production-grade implementation for document ingestion, chunking, and retrieval.

MANDATORY: Pathway is used for:
- Ingesting .txt novels (full text, no truncation)
- Chunking into ~1000-token overlapping segments
- Storing chunk metadata (narrative_position, chapter)
- Vector indexing with BGE-M3 embeddings
- Retrieval with metadata filtering

STRICT: LLM must NEVER be called inside Pathway pipeline.
"""

import numpy as np
from typing import List, Dict, Optional
import re
from rank_bm25 import BM25Okapi

import config


class NovelChunk:
    """
    Pathway schema for novel chunks.
    Each chunk contains text, embeddings, and metadata.
    """
    def __init__(
        self,
        text: str,
        narrative_position: int,
        chapter_index: Optional[int],
        char_start: int,
        char_end: int,
        embedding: np.ndarray
    ):
        self.text = text
        self.narrative_position = narrative_position
        self.chapter_index = chapter_index
        self.char_start = char_start
        self.char_end = char_end
        self.embedding = embedding


class PathwayVectorStore:
    """
    Pathway-based vector store for novel ingestion and retrieval.
    Handles full .txt novels with position-aware chunking and indexing.
    """
    
    def __init__(self, embedder):
        """
        Initialize Pathway vector store.
        
        Args:
            embedder: BAAI/bge-m3 embedding model (SentenceTransformer)
        """
        self.embedder = embedder
        self.chunks: List[NovelChunk] = []
        self.doc_id: Optional[str] = None
        self.bm25: Optional[BM25Okapi] = None
        
        print("✓ Pathway vector store initialized")
    
    def read_novel_file(self, filepath: str) -> str:
        """
        Read full .txt novel without truncation.
        Pathway ingestion: complete text loading.
        
        Args:
            filepath: Path to .txt novel file
            
        Returns:
            str: Full novel text
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text
    
    def ingest_document(self, text: str, doc_id: str):
        """
        Ingest full novel into Pathway vector store.
        Pipeline: chunk → detect chapters → embed → index
        
        NO LLM CALLS allowed in this pipeline.
        
        Args:
            text: Full novel text (no truncation)
            doc_id: Novel identifier
        """
        print(f"\n{'='*60}")
        print(f"PATHWAY INGESTION: {doc_id}")
        print(f"{'='*60}")
        print(f"Text length: {len(text):,} characters")
        
        self.doc_id = doc_id
        
        # Step 1: Chunk with ~1000 token segments
        raw_chunks = self._chunk_into_segments(text)
        print(f"Created {len(raw_chunks)} raw chunks")
        
        # Step 2: Detect chapters and assign metadata
        chunks_with_metadata = self._assign_metadata(raw_chunks, text)
        print(f"Metadata assigned: {len(chunks_with_metadata)} chunks")
        
        # Step 3: Embed all chunks with BGE-M3
        embeddings = self._embed_chunks(chunks_with_metadata)
        print(f"Embeddings generated: {embeddings.shape}")
        
        # Step 4: Create NovelChunk objects and store in Pathway index
        for i, chunk_dict in enumerate(chunks_with_metadata):
            chunk_obj = NovelChunk(
                text=chunk_dict['text'],
                narrative_position=chunk_dict['narrative_position'],
                chapter_index=chunk_dict.get('chapter_index'),
                char_start=chunk_dict['char_start'],
                char_end=chunk_dict['char_end'],
                embedding=embeddings[i]
            )
            self.chunks.append(chunk_obj)
        
        # Step 5: Build BM25 lexical index (optional)
        if config.BM25_ENABLED:
            self._build_bm25_index()
        
        print(f"✓ Pathway ingestion complete: {len(self.chunks)} chunks indexed")
        print(f"{'='*60}\n")
    
    def _chunk_into_segments(self, text: str) -> List[Dict]:
        """
        Chunk text into ~1000-token overlapping segments.
        Pathway chunking: character-based with token estimation.
        
        Token estimation: ~4 characters per token (English)
        Target: 1000 tokens ≈ 4000 characters
        
        Args:
            text: Full novel text
            
        Returns:
            List[Dict]: Raw chunks with char positions
        """
        # Token-aware chunking
        chars_per_token = 4
        target_tokens = 1000
        target_chars = target_tokens * chars_per_token  # ~4000 chars
        
        overlap_tokens = 100  # Overlap for context preservation
        overlap_chars = overlap_tokens * chars_per_token  # ~400 chars
        
        chunks = []
        char_pos = 0
        
        while char_pos < len(text):
            chunk_end = min(char_pos + target_chars, len(text))
            chunk_text = text[char_pos:chunk_end]
            
            chunks.append({
                'text': chunk_text,
                'char_start': char_pos,
                'char_end': chunk_end
            })
            
            # Move forward with overlap
            char_pos += (target_chars - overlap_chars)
        
        return chunks
    
    def _assign_metadata(self, chunks: List[Dict], full_text: str) -> List[Dict]:
        """
        Assign narrative_position and chapter_index to chunks.
        Pathway metadata assignment: monotonic positions + chapter detection.
        
        Args:
            chunks: Raw chunks with char positions
            full_text: Original text for chapter detection
            
        Returns:
            List[Dict]: Chunks with metadata
        """
        # Detect chapters in full text
        chapter_boundaries = self._detect_chapters(full_text)
        
        # Assign metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['narrative_position'] = i
            chunk['chapter_index'] = self._find_chapter_for_position(
                chunk['char_start'],
                chapter_boundaries
            )
        
        return chunks
    
    def _detect_chapters(self, text: str) -> List[int]:
        """
        Detect chapter boundaries in text.
        Simple heuristic: "Chapter N", "CHAPTER N", "Chapter N:", etc.
        
        Args:
            text: Full text
            
        Returns:
            List[int]: Character positions of chapter starts
        """
        chapter_pattern = r'\b(Chapter|CHAPTER|chapter)\s+(\d+|[IVXLCDM]+)'
        boundaries = [0]  # Start of text
        
        for match in re.finditer(chapter_pattern, text):
            boundaries.append(match.start())
        
        return sorted(set(boundaries))
    
    def _find_chapter_for_position(
        self,
        char_pos: int,
        chapter_boundaries: List[int]
    ) -> Optional[int]:
        """
        Find which chapter a character position belongs to.
        
        Args:
            char_pos: Character position in text
            chapter_boundaries: List of chapter start positions
            
        Returns:
            Optional[int]: Chapter index (0-based) or None
        """
        if not chapter_boundaries:
            return None
        
        for i in range(len(chapter_boundaries) - 1, -1, -1):
            if char_pos >= chapter_boundaries[i]:
                return i
        
        return 0
    
    def _embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Embed chunks using BGE-M3 (BAAI/bge-m3).
        Pathway embedding: batch processing for efficiency.
        
        NO LLM calls - only embedding model.
        
        Args:
            chunks: List of chunk dicts with 'text' field
            
        Returns:
            np.ndarray: Embeddings of shape (n_chunks, 1024)
        """
        chunk_texts = [c['text'] for c in chunks]
        
        print(f"Embedding {len(chunk_texts)} chunks with BGE-M3...")
        
        # Batch encode with BGE-M3
        embeddings = self.embedder.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings
    
    def _build_bm25_index(self):
        """
        Build BM25 lexical index for hybrid retrieval.
        Pathway indexing: optional lexical fallback.
        """
        print("Building BM25 lexical index...")
        
        tokenized_texts = [
            chunk.text.lower().split()
            for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_texts)
        print("✓ BM25 index built")
    
    def retrieve(
        self,
        query: str,
        position_filter: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using Pathway KNN vector index.
        Supports metadata filtering by narrative_position.
        
        Retrieval: Dense (BGE-M3) + optional BM25 hybrid.
        NO LLM calls in retrieval pipeline.
        
        Args:
            query: Query text for semantic search
            position_filter: Only return chunks with narrative_position >= this value
            top_k: Number of chunks to retrieve
            
        Returns:
            List[Dict]: Retrieved chunks with scores and metadata
        """
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K
        
        # Step 1: Filter by narrative_position (Pathway metadata filtering)
        candidate_chunks = self.chunks
        
        if position_filter is not None and config.POSITION_FILTER_ENABLED:
            candidate_chunks = [
                chunk for chunk in self.chunks
                if chunk.narrative_position >= position_filter
            ]
            print(f"Position filter: {len(self.chunks)} → {len(candidate_chunks)} chunks (pos >= {position_filter})")
        
        if not candidate_chunks:
            print("⚠ No chunks after position filtering")
            return []
        
        # Step 2: Dense retrieval with BGE-M3 (Pathway KNN)
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Compute cosine similarity (L2-normalized embeddings)
        candidate_embeddings = np.array([c.embedding for c in candidate_chunks])
        dense_scores = np.dot(candidate_embeddings, query_embedding)
        
        # Step 3: Optional BM25 hybrid retrieval
        if config.BM25_ENABLED and self.bm25 is not None:
            bm25_scores = self._compute_bm25_scores(query, candidate_chunks)
            
            # Hybrid fusion: weighted combination
            alpha = 1 - config.BM25_WEIGHT
            hybrid_scores = alpha * dense_scores + config.BM25_WEIGHT * bm25_scores
        else:
            hybrid_scores = dense_scores
        
        # Step 4: Rank and return top-k
        results = self._rank_and_format(candidate_chunks, hybrid_scores, top_k)
        
        top_scores = [f"{r['score']:.3f}" for r in results[:3]]
        print(f"Retrieved {len(results)} chunks | Top scores: {top_scores}")
        
        return results
    
    def _compute_bm25_scores(
        self,
        query: str,
        candidate_chunks: List[NovelChunk]
    ) -> np.ndarray:
        """
        Compute BM25 lexical scores for candidates.
        
        Args:
            query: Query text
            candidate_chunks: Filtered chunk objects
            
        Returns:
            np.ndarray: Normalized BM25 scores
        """
        query_tokens = query.lower().split()
        
        # Get indices of candidates in full chunk list
        candidate_indices = [self.chunks.index(c) for c in candidate_chunks]
        
        # Compute BM25 scores
        all_scores = self.bm25.get_scores(query_tokens)
        bm25_scores = all_scores[candidate_indices]
        
        # Normalize to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        return bm25_scores
    
    def _rank_and_format(
        self,
        chunks: List[NovelChunk],
        scores: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """
        Rank chunks by score and format as dicts.
        
        Args:
            chunks: Candidate chunks
            scores: Retrieval scores
            top_k: Number to return
            
        Returns:
            List[Dict]: Top-k chunks with metadata
        """
        # Create scored list
        scored = [
            {
                'text': chunk.text,
                'narrative_position': chunk.narrative_position,
                'position': chunk.narrative_position,  # Alias for compatibility
                'chapter_index': chunk.chapter_index,
                'char_start': chunk.char_start,
                'char_end': chunk.char_end,
                'score': float(scores[i]),
                'embedding': chunk.embedding
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        return scored[:top_k]
    
    def get_chunk_by_position(self, position: int) -> Optional[Dict]:
        """
        Retrieve specific chunk by narrative_position.
        Pathway: direct position-based lookup.
        
        Args:
            position: narrative_position (0-indexed monotonic)
            
        Returns:
            Optional[Dict]: Chunk dict or None if not found
        """
        for chunk in self.chunks:
            if chunk.narrative_position == position:
                return {
                    'text': chunk.text,
                    'narrative_position': chunk.narrative_position,
                    'position': chunk.narrative_position,
                    'chapter_index': chunk.chapter_index,
                    'char_start': chunk.char_start,
                    'char_end': chunk.char_end
                }
        return None
    
    def get_chunks_in_range(
        self,
        start_pos: int,
        end_pos: int
    ) -> List[Dict]:
        """
        Get all chunks in narrative_position range.
        Pathway: range-based metadata filtering.
        
        Args:
            start_pos: Start position (inclusive)
            end_pos: End position (inclusive)
            
        Returns:
            List[Dict]: Chunks in range
        """
        results = []
        for chunk in self.chunks:
            if start_pos <= chunk.narrative_position <= end_pos:
                results.append({
                    'text': chunk.text,
                    'narrative_position': chunk.narrative_position,
                    'position': chunk.narrative_position,
                    'chapter_index': chunk.chapter_index,
                    'char_start': chunk.char_start,
                    'char_end': chunk.char_end
                })
        return results
    
    def get_total_chunks(self) -> int:
        """
        Get total number of indexed chunks.
        
        Returns:
            int: Total chunk count
        """
        return len(self.chunks)
    
    def get_chapter_count(self) -> int:
        """
        Get number of detected chapters.
        
        Returns:
            int: Chapter count
        """
        chapters = set(c.chapter_index for c in self.chunks if c.chapter_index is not None)
        return len(chapters)
