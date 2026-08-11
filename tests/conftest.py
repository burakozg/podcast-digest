"""Shared fixtures.

No test touches the network or a real CouchDB: storage is MemoryStore, HTTP is
respx-mocked, and the LLM is a fake at the ``complete_structured`` boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import FakeLLM, make_episode, make_settings

from podcast_agent import net
from podcast_agent.api.auth import reset_throttle
from podcast_agent.config import Settings
from podcast_agent.db import MemoryStore


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip PODAGENT_* so a developer's shell or .env cannot change what the
    tests assert.

    Clearing os.environ is not enough on its own: pydantic-settings reads `.env`
    from disk as its own source, and tests run from the repo root where a real
    one sits. A developer with PODAGENT_OPENROUTER_API_KEY in that file saw
    `test_cloud_endpoint_without_a_key_is_rejected` fail while CI passed — the
    suite was quietly asserting against their credentials.
    """
    for key in list(os.environ):
        if key.startswith("PODAGENT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every hostname resolves to one public address unless a test says otherwise.

    The URL guard resolves each redirect hop and refuses private answers. Left
    alone that would put a real DNS lookup in the middle of the suite: slow,
    dependent on someone else's zone, and — for a host that happens not to
    exist — a rejection that has nothing to do with what is being tested.

    Tests about the address check patch :func:`podcast_agent.net._resolve`
    themselves; everything else gets a boring public answer.
    """

    async def _public(host: str) -> list[str]:
        return ["93.184.216.34"]

    monkeypatch.setattr(net, "_resolve", _public)


@pytest.fixture(autouse=True)
def _forget_failed_auth() -> None:
    """The auth throttle counts failures per address, in module state.

    Left alone it would leak between tests: a file that exercises a few 401s
    would make a later test's 401 a 429, and which test broke would depend on
    the order they ran in.
    """
    reset_throttle()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def episode() -> dict[str, Any]:
    return make_episode()
