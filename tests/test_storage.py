"""Tests for the ChromaDB storage layer."""

from __future__ import annotations

import json

import pytest

from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.storage.embeddings import deterministic_embedding


class TestVoiceStoreSamples:
    """Test writing-sample CRUD operations."""

    def test_add_and_count(self, store: VoiceStore) -> None:
        chunks = ["First chunk of writing.", "Second chunk of writing."]
        embeddings = [deterministic_embedding(c) for c in chunks]
        ids = store.add_samples(
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": "blog", "title": "Test Post"},
        )
        assert len(ids) == 2
        assert store.total_samples == 2

    def test_query_returns_similar(self, store: VoiceStore) -> None:
        chunks = [
            "React hooks simplify state management in functional components.",
            "Python asyncio enables concurrent programming patterns.",
            "CSS grid provides powerful two-dimensional layout capabilities.",
        ]
        embeddings = [deterministic_embedding(c) for c in chunks]
        store.add_samples(
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": "blog", "title": "Tech Notes"},
        )

        query_emb = deterministic_embedding("React state management")
        results = store.query_samples(query_embedding=query_emb, top_k=2)
        assert len(results) == 2
        assert all("content" in r for r in results)
        assert all("similarity" in r for r in results)

    def test_query_with_source_filter(self, store: VoiceStore) -> None:
        store.add_samples(
            chunks=["Blog content here."],
            embeddings=[deterministic_embedding("Blog content here.")],
            metadata={"source": "blog"},
        )
        store.add_samples(
            chunks=["Social media post."],
            embeddings=[deterministic_embedding("Social media post.")],
            metadata={"source": "social"},
        )

        query_emb = deterministic_embedding("content")
        blog_results = store.query_samples(query_embedding=query_emb, top_k=5, source_filter="blog")
        assert all(r["source"] == "blog" for r in blog_results)

    def test_list_samples_pagination(self, store: VoiceStore) -> None:
        for i in range(5):
            store.add_samples(
                chunks=[f"Sample number {i}"],
                embeddings=[deterministic_embedding(f"Sample number {i}")],
                metadata={"source": "blog", "title": f"Post {i}"},
            )

        entries, total = store.list_samples(limit=2, offset=0)
        assert total == 5
        assert len(entries) == 2

    def test_sources_breakdown(self, store: VoiceStore) -> None:
        store.add_samples(
            chunks=["Blog one", "Blog two"],
            embeddings=[deterministic_embedding(c) for c in ["Blog one", "Blog two"]],
            metadata={"source": "blog"},
        )
        store.add_samples(
            chunks=["Email one"],
            embeddings=[deterministic_embedding("Email one")],
            metadata={"source": "email"},
        )

        breakdown = store.sources_breakdown()
        assert breakdown["blog"] == 2
        assert breakdown["email"] == 1

    def test_empty_store(self, store: VoiceStore) -> None:
        assert store.total_samples == 0
        entries, total = store.list_samples()
        assert total == 0
        assert entries == []

    def test_get_corpus_excerpts_respects_limits(self, store: VoiceStore) -> None:
        long_a = "Alpha sentence. " * 200
        long_b = "Beta sentence. " * 200
        store.add_samples(
            chunks=[long_a, long_b],
            embeddings=[deterministic_embedding(long_a), deterministic_embedding(long_b)],
            metadata={"source": "blog"},
        )
        excerpts = store.get_corpus_excerpts(
            max_chunks=5,
            max_chars_per_chunk=100,
            max_total_chars=150,
        )
        assert len(excerpts) >= 1
        assert sum(len(e) for e in excerpts) <= 150
        assert all(len(e) <= 100 for e in excerpts)


class TestVoiceStoreProfile:
    """Test voice-profile storage operations."""

    def test_save_and_get_learned_style(self, store: VoiceStore) -> None:
        style_data = {
            "avg_sentence_length": 14.2,
            "vocabulary_richness": 0.72,
            "formality_score": 0.45,
            "dominant_tone": "conversational",
            "rhetorical_patterns": ["uses analogies", "short punchy sentences"],
        }
        store.save_learned_style(style_data)
        retrieved = store.get_learned_style()
        assert retrieved is not None
        assert retrieved["dominant_tone"] == "conversational"
        assert retrieved["avg_sentence_length"] == 14.2

    def test_save_and_get_guidelines(self, store: VoiceStore) -> None:
        guidelines = {
            "pillars": ["practical over theoretical", "teach through building"],
            "tone": {"formality": 0.4, "humor": 0.3, "technical_depth": 0.8, "warmth": 0.6},
            "preferred_vocabulary": ["ship", "build", "scale"],
            "avoided_vocabulary": ["synergy", "leverage", "paradigm"],
        }
        store.save_guidelines(guidelines)
        retrieved = store.get_guidelines()
        assert retrieved is not None
        assert "practical over theoretical" in retrieved["pillars"]
        assert "synergy" in retrieved["avoided_vocabulary"]

    def test_update_guidelines_overwrites(self, store: VoiceStore) -> None:
        store.save_guidelines({"pillars": ["old pillar"]})
        store.save_guidelines({"pillars": ["new pillar"], "topics": ["AI"]})
        retrieved = store.get_guidelines()
        assert retrieved["pillars"] == ["new pillar"]
        assert retrieved["topics"] == ["AI"]

    def test_get_nonexistent_returns_none(self, store: VoiceStore) -> None:
        assert store.get_learned_style() is None
        assert store.get_guidelines() is None

    def test_profile_last_updated(self, store: VoiceStore) -> None:
        assert store.get_profile_last_updated() is None
        store.save_learned_style({"dominant_tone": "test"})
        ts = store.get_profile_last_updated()
        assert ts is not None
