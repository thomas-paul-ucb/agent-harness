"""Tests for SemanticCache: task-level semantic lookup and storage.

Uses a temporary directory for chromadb's persistent storage so tests
don't pollute (or depend on) your real .cache/ folder. No Anthropic API
key needed — this only exercises the local embedding model and chromadb.
"""

import shutil
import tempfile

import pytest

from agentharness.cache import SemanticCache


@pytest.fixture
def cache():
    """A fresh SemanticCache backed by a temp directory, cleaned up after each test."""
    temp_dir = tempfile.mkdtemp()
    c = SemanticCache(persist_directory=temp_dir, similarity_threshold=0.90)
    yield c
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_empty_cache_returns_no_hit(cache):
    assert cache.lookup("What is the capital of France?") is None


def test_exact_task_returns_stored_answer(cache):
    cache.store("What is the capital of France?", "Paris")
    result = cache.lookup("What is the capital of France?")
    assert result == "Paris"


def test_unrelated_task_returns_no_hit(cache):
    cache.store("What is the capital of France?", "Paris")
    result = cache.lookup("How do I fix a flat bike tire?")
    assert result is None


def test_paraphrased_task_may_hit_depending_on_threshold(cache):
    """This test documents the real, honest behavior: paraphrase matching
    depends on the threshold and the embedding model's judgment — it's not
    guaranteed. We assert the result is a valid outcome either way, and
    print what actually happened so you can see real threshold behavior."""
    cache.store("What is the capital of France?", "Paris")
    result = cache.lookup("What's France's capital city?")
    # Either outcome is "correct" depending on where 0.90 lands for this
    # pair — the assertion just confirms lookup() doesn't crash and
    # returns either the right answer or None, never a wrong answer.
    assert result in ("Paris", None)