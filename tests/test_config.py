"""Configuration tests (§8): invalid config must crash loudly, never start half-configured."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from helpers import DROP, make_settings
from pydantic import ValidationError

import podcast_agent.config as config_module
from podcast_agent.config import LLMTierConfig, PipelineConfig, Provider, Settings, load_settings
from podcast_agent.settings_store import (
    OverrideRejected,
    allowed_api_base_hosts,
    check_api_bases,
)


def build(tmp_path: Path, **overrides: Any) -> Settings:
    return make_settings(tmp_path, **overrides)


class TestValidConfig:
    def test_repo_config_yaml_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The shipped config.yaml must load — it is the deployment default."""
        monkeypatch.setenv("PODAGENT_ADMIN_API_KEY", "k")
        monkeypatch.setenv("PODAGENT_COUCHDB_PASSWORD", "p")
        monkeypatch.setenv("PODAGENT_OPENROUTER_API_KEY", "o")
        # Both providers are in the shipped chains while there is no local model
        # host: OpenRouter primary, Anthropic fallback. Omitting either key here
        # would fail this test for the same reason a deployment would refuse to
        # boot, which is the behaviour, not a test-setup detail.
        monkeypatch.setenv("PODAGENT_ANTHROPIC_API_KEY", "a")
        settings = load_settings(Path(__file__).parent.parent / "config.yaml")
        assert len(settings.podcasts) == 14
        assert {i.key for i in settings.interest_profile} >= {"ot_ics", "ai_agent_security"}
        assert settings.scheduler.timezone == "Europe/Stockholm"

    def test_generated_local_config_needs_no_anthropic_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dev machine must boot with the OpenRouter key alone.

        Adding the Anthropic fallback to config.yaml made the *local* config
        unbootable: an anthropic endpoint in the chain is a hard startup failure
        when PODAGENT_ANTHROPIC_API_KEY is unset, so `uv run podcast-agent`
        died on a machine that had never needed a second vendor's key. This
        pins the property that broke — the generator strips those endpoints —
        rather than the mechanism, which is free to change.
        """
        import importlib.util

        root = Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "make_local_config", root / "scripts" / "make-local-config.py"
        )
        assert spec is not None and spec.loader is not None
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)

        text = (root / "config.yaml").read_text(encoding="utf-8")
        for old, new, _must_match in generator.REPLACEMENTS:
            text = text.replace(old, new)
        text, dropped = generator.strip_anthropic_fallbacks(text)
        assert dropped >= 1, "expected the generator to strip anthropic endpoints"

        local = tmp_path / "config.local.yaml"
        local.write_text(text, encoding="utf-8")

        monkeypatch.setenv("PODAGENT_ADMIN_API_KEY", "k")
        monkeypatch.setenv("PODAGENT_COUCHDB_PASSWORD", "p")
        monkeypatch.setenv("PODAGENT_OPENROUTER_API_KEY", "o")
        monkeypatch.delenv("PODAGENT_ANTHROPIC_API_KEY", raising=False)

        settings = load_settings(local)
        for name, tier in settings.llm.tiers.items():
            assert tier.active_chain(), f"tier {name} has no reachable endpoint"
            assert all(e.provider is not Provider.ANTHROPIC for e in tier.active_chain()), (
                f"tier {name} still has an anthropic endpoint, so local dev needs that key"
            )

    def test_repo_config_has_no_local_only_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every shipped tier must have a reachable endpoint without a model host.

        The interim configuration is entirely cloud, so the guard that used to
        protect data sovereignty (`allow_cloud_fallback: false`) would now empty
        a chain instead. This pins the property that matters — every tier can
        actually run — rather than the provider names, which are expected to
        change back once there is a machine to run Ollama on.
        """
        monkeypatch.setenv("PODAGENT_ADMIN_API_KEY", "k")
        monkeypatch.setenv("PODAGENT_COUCHDB_PASSWORD", "p")
        monkeypatch.setenv("PODAGENT_OPENROUTER_API_KEY", "o")
        monkeypatch.setenv("PODAGENT_ANTHROPIC_API_KEY", "a")
        settings = load_settings(Path(__file__).parent.parent / "config.yaml")
        for name, tier in settings.llm.tiers.items():
            assert tier.active_chain(), f"tier {name} has no reachable endpoint"

    def test_env_var_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODAGENT_PIPELINE__DIGEST_THRESHOLD", "6")
        settings = build(tmp_path)
        assert settings.pipeline.digest_threshold == 6

    def test_env_override_still_gets_cross_field_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raising digest_threshold above top_pick_threshold is incoherent, so an
        env override must fail as loudly as a bad config.yaml would."""
        monkeypatch.setenv("PODAGENT_PIPELINE__DIGEST_THRESHOLD", "9")
        with pytest.raises(ValidationError, match="top_pick_threshold"):
            build(tmp_path)

    def test_secrets_are_not_stringified(self, tmp_path: Path) -> None:
        """SecretStr keeps keys out of logs and reprs (§8)."""
        settings = build(tmp_path)
        assert "test-admin-key" not in repr(settings)
        assert settings.admin_api_key is not None
        assert settings.admin_api_key.get_secret_value() == "test-admin-key"


class TestRejectsBadConfig:
    def test_duplicate_podcast_slug(self, tmp_path: Path) -> None:
        podcasts = [
            {"slug": "dup", "name": "A", "feed_url": "https://a.example/f"},
            {"slug": "dup", "name": "B", "feed_url": "https://b.example/f"},
        ]
        with pytest.raises(ValidationError, match="duplicate podcast slug"):
            build(tmp_path, podcasts=podcasts)

    def test_duplicate_interest_key(self, tmp_path: Path) -> None:
        profile = [
            {"key": "k", "label": "A", "description": "d", "weight": 5},
            {"key": "k", "label": "B", "description": "d", "weight": 5},
        ]
        with pytest.raises(ValidationError, match="duplicate interest key"):
            build(tmp_path, interest_profile=profile)

    def test_empty_interest_profile(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="at least one interest"):
            build(tmp_path, interest_profile=[])

    def test_no_enabled_podcasts(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="no enabled podcasts"):
            build(
                tmp_path,
                podcasts=[
                    {
                        "slug": "off",
                        "name": "Off",
                        "feed_url": "https://o.example/f",
                        "enabled": False,
                    }
                ],
            )

    def test_invalid_cron_expression(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="invalid cron expression"):
            build(tmp_path, scheduler={"digest_cron": "not a cron"})

    def test_unknown_timezone(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="unknown timezone"):
            build(tmp_path, scheduler={"timezone": "Mars/Olympus_Mons"})

    def test_non_http_feed_url(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="http"):
            build(
                tmp_path,
                podcasts=[{"slug": "s", "name": "S", "feed_url": "file:///etc/passwd"}],
            )

    def test_typo_in_key_is_rejected(self, tmp_path: Path) -> None:
        """extra='forbid' turns a silent no-op setting into a startup failure."""
        with pytest.raises(ValidationError):
            build(tmp_path, pipeline={"digest_threshhold": 5})

    def test_missing_required_tier(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="missing required tier"):
            build(
                tmp_path,
                llm={"tiers": {"tier0": {"primary": {"provider": "ollama", "model": "m"}}}},
            )

    def test_thresholds_must_be_ordered(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="t_rel_low"):
            build(tmp_path, pipeline={"t_rel_low": 9, "t_rel_high": 3})

    def test_top_pick_threshold_below_digest_threshold(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="top_pick_threshold"):
            build(tmp_path, pipeline={"digest_threshold": 8, "top_pick_threshold": 5})

    def test_score_out_of_range(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            build(tmp_path, pipeline={"digest_threshold": 42})

    def test_remote_asr_requires_url(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match=r"asr\.remote_url is required"):
            build(tmp_path, asr={"backend": "remote"})


class TestCloudCredentialGuard:
    """A configured cloud endpoint with no key would fail silently at 3am (§8)."""

    def _with_openrouter(self) -> dict[str, Any]:
        return {
            "tiers": {
                "tier0": {
                    "primary": {"provider": "ollama", "model": "local"},
                    "fallbacks": [{"provider": "openrouter", "model": "remote/model"}],
                },
                "tier1": {"primary": {"provider": "ollama", "model": "local-big"}},
            }
        }

    def test_openrouter_without_key_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="PODAGENT_OPENROUTER_API_KEY"):
            build(tmp_path, llm=self._with_openrouter(), openrouter_api_key=None)

    def test_openrouter_with_key_is_accepted(self, tmp_path: Path) -> None:
        settings = build(tmp_path, llm=self._with_openrouter(), openrouter_api_key="sk-test")
        assert settings.api_key_for(Provider.OPENROUTER) == "sk-test"

    def test_disabling_cloud_fallback_removes_the_requirement(self, tmp_path: Path) -> None:
        """allow_cloud_fallback: false must be a real switch, not a suggestion (§10.6)."""
        llm = self._with_openrouter()
        llm["tiers"]["tier0"]["allow_cloud_fallback"] = False
        settings = build(tmp_path, llm=llm, openrouter_api_key=None)
        chain = settings.llm.tiers["tier0"].active_chain()
        assert len(chain) == 1
        assert chain[0].provider is Provider.OLLAMA

    def test_local_only_tier_with_no_local_endpoint_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="could never run"):
            LLMTierConfig(
                primary={"provider": "openrouter", "model": "m"},  # type: ignore[arg-type]
                allow_cloud_fallback=False,
            )


class TestModelStrings:
    def test_ollama_maps_to_chat_endpoint(self, tmp_path: Path) -> None:
        """JSON-mode structured output needs /api/chat, not /api/generate."""
        settings = build(tmp_path)
        endpoint = settings.llm.tiers["tier0"].primary
        assert endpoint.litellm_model() == "ollama_chat/test-small"

    def test_ollama_gets_a_default_base_url(self, tmp_path: Path) -> None:
        settings = build(tmp_path)
        assert settings.llm.tiers["tier0"].primary.resolved_api_base() == "http://localhost:11434"

    def test_cloud_endpoint_has_no_default_base_url(self, tmp_path: Path) -> None:
        settings = build(
            tmp_path,
            llm={
                "tiers": {
                    "tier0": {"primary": {"provider": "openrouter", "model": "x/y"}},
                    "tier1": {"primary": {"provider": "ollama", "model": "z"}},
                }
            },
            openrouter_api_key="sk",
        )
        assert settings.llm.tiers["tier0"].primary.resolved_api_base() is None


class TestPipelineDefaultsMatchDesignDoc:
    def test_default_thresholds(self) -> None:
        cfg = PipelineConfig()
        assert (cfg.t_conf_high, cfg.t_rel_low, cfg.t_rel_high) == (7, 4, 7)
        assert cfg.digest_threshold == 5
        assert cfg.initial_lookback_days == 14
        assert cfg.max_retries == 3


def test_missing_config_file_uses_defaults(tmp_path: Path) -> None:
    """A missing config file is not a crash; the required fields still are."""
    config_module._active_yaml_path = tmp_path / "absent.yaml"
    with pytest.raises(ValidationError):
        Settings()  # llm has no default → clear error rather than a broken boot


class TestDotEnvSharedWithCompose:
    """The .env file is shared with docker-compose, which needs its own
    non-prefixed variables. Regression: the default dotenv source passed those
    through to the model and extra="forbid" rejected them, so a correctly-filled
    .env crashed the app at startup — following the README broke the deploy.
    """

    @pytest.fixture(autouse=True)
    def _read_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Undo conftest's global disabling: this class is *about* .env loading.

        Everywhere else the file is switched off so the developer's real one
        cannot reach the suite; here each test writes its own into tmp_path and
        chdirs to it, so reading it back is the whole point.
        """
        monkeypatch.setitem(Settings.model_config, "env_file", ".env")

    def _write_env(self, tmp_path: Path, body: str) -> Path:
        env_file = tmp_path / ".env"
        env_file.write_text(body)
        return env_file

    def test_shipped_env_example_boots_the_app(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Copying .env.example to .env and filling it in must work — it is the
        documented first step in the README."""
        example = (Path(__file__).parent.parent / ".env.example").read_text()
        filled = example.replace(
            "PODAGENT_ADMIN_API_KEY=", "PODAGENT_ADMIN_API_KEY=abc123"
        ).replace("COUCHDB_PASSWORD=", "COUCHDB_PASSWORD=secret")
        self._write_env(tmp_path, filled)
        monkeypatch.chdir(tmp_path)

        # Omit the test default so the key must come from the .env file itself.
        settings = build(tmp_path, admin_api_key=DROP)
        assert settings.admin_api_key is not None
        assert settings.admin_api_key.get_secret_value() == "abc123"

    def test_compose_only_vars_are_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_env(
            tmp_path,
            "PODAGENT_ADMIN_API_KEY=k\n"
            "COUCHDB_USER=podagent\n"
            "DIGEST_DIR=./data/digests\n"
            "MACVLAN_PARENT=eth0\n"
            "SOME_UNRELATED_SHELL_VAR=1\n",
        )
        monkeypatch.chdir(tmp_path)
        settings = build(tmp_path, admin_api_key=DROP)
        assert settings.admin_api_key is not None
        assert not hasattr(settings, "couchdb_user")

    def test_prefixed_vars_in_dotenv_still_apply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_env(
            tmp_path,
            "PODAGENT_ADMIN_API_KEY=k\nPODAGENT_COUCHDB_PASSWORD=pw\n",
        )
        monkeypatch.chdir(tmp_path)
        settings = build(tmp_path, admin_api_key=DROP)
        assert settings.couchdb_password is not None
        assert settings.couchdb_password.get_secret_value() == "pw"

    def test_a_typo_in_config_yaml_is_still_fatal(self, tmp_path: Path) -> None:
        """Tolerating compose vars must not weaken validation of real config."""
        with pytest.raises(ValidationError):
            build(tmp_path, pipeline={"digest_threshhold": 5})


class TestOverrideMerging:
    """A stored override must not inherit the baseline's other provider.

    The console stored an OpenRouter primary while `config.yaml` declared an
    Ollama one. Merging them key by key handed OpenRouter Ollama's `api_base`
    and its `num_ctx`/`num_predict` — parameters OpenRouter ignores, so tier-1
    ran with no reply limit and truncated every reply mid-JSON. Billed, useless,
    and it fell through to the local model anyway.
    """

    def _base(self) -> dict:
        return {
            "llm": {
                "log_llm_io": False,
                "tiers": {
                    "tier1": {
                        "timeout_s": 900,
                        "primary": {
                            "provider": "ollama",
                            "model": "qwen3.5:9b",
                            "api_base": "http://127.0.0.1:11434",
                            "extra_params": {"num_ctx": 8192, "num_predict": 1600},
                        },
                    }
                },
            }
        }

    def test_a_tier_replaces_rather_than_merges(self) -> None:
        from podcast_agent.settings_store import deep_merge

        merged = deep_merge(
            self._base(),
            {
                "llm": {
                    "tiers": {
                        "tier1": {
                            "timeout_s": 900,
                            "primary": {
                                "provider": "openrouter",
                                "model": "qwen/qwen3-32b",
                                "api_base": None,
                                "max_tokens": 2000,
                                "extra_params": {},
                            },
                        }
                    }
                }
            },
        )
        primary = merged["llm"]["tiers"]["tier1"]["primary"]
        assert primary["provider"] == "openrouter"
        # None of the Ollama endpoint survives underneath it.
        assert primary["api_base"] is None
        assert primary["extra_params"] == {}
        assert primary["max_tokens"] == 2000

    def test_a_tier_the_override_omits_keeps_its_baseline(self) -> None:
        """Replacement is per tier, not of the whole `tiers` map."""
        from podcast_agent.settings_store import deep_merge

        merged = deep_merge(
            {"llm": {"tiers": {"tier0": {"timeout_s": 60}, "tier1": {"timeout_s": 900}}}},
            {"llm": {"tiers": {"tier1": {"timeout_s": 300}}}},
        )
        assert merged["llm"]["tiers"]["tier0"]["timeout_s"] == 60
        assert merged["llm"]["tiers"]["tier1"]["timeout_s"] == 300

    def test_sibling_llm_keys_still_merge(self) -> None:
        """Only tiers replace; `log_llm_io` beside them must be left alone."""
        from podcast_agent.settings_store import deep_merge

        merged = deep_merge(self._base(), {"llm": {"tiers": {"tier1": {"timeout_s": 300}}}})
        assert merged["llm"]["log_llm_io"] is False

    def test_everything_else_still_merges_key_by_key(self) -> None:
        from podcast_agent.settings_store import deep_merge

        merged = deep_merge(
            {"pipeline": {"digest_threshold": 5, "t_rel_low": 4}},
            {"pipeline": {"digest_threshold": 6}},
        )
        assert merged["pipeline"] == {"digest_threshold": 6, "t_rel_low": 4}

    def test_lists_still_replace(self) -> None:
        from podcast_agent.settings_store import deep_merge

        merged = deep_merge({"interest_profile": [1, 2, 3]}, {"interest_profile": [9]})
        assert merged["interest_profile"] == [9]


class TestShippedAllowlist:
    """The allowlist is the difference between a show working and vanishing.

    A rejected enclosure discards the whole episode (`Ingestor._ingest_entry`),
    so a missing prefixer is not a degraded transcript — it is a podcast that
    silently ingests nothing. Click Here did exactly that for a day.
    """

    def _allowlist(self) -> set[str]:
        import yaml

        text = (Path(__file__).parent.parent / "config.yaml").read_text()
        return set(yaml.safe_load(text)["security"]["cdn_allowlist"])

    def test_every_link_in_a_prefix_chain_is_present(self) -> None:
        """A chain is only as usable as its outermost host.

        Click Here's audio is swap.fm -> mgln.ai -> podtrac -> prxu.org, and the
        guard only ever sees the first of those.
        """
        assert {"swap.fm", "mgln.ai", "podtrac.com", "prxu.org"} <= self._allowlist()

    def test_the_local_config_inherits_it(self) -> None:
        """config.local.yaml is generated; a hand-edit there would be lost."""
        import yaml

        path = Path(__file__).parent.parent / "config.local.yaml"
        if not path.exists():  # not present in a clean checkout
            pytest.skip("config.local.yaml is generated and gitignored")
        local = set(yaml.safe_load(path.read_text())["security"]["cdn_allowlist"])
        assert self._allowlist() <= local

    def test_a_prefixed_enclosure_is_permitted(self) -> None:
        from podcast_agent.config import SecurityConfig
        from podcast_agent.net import UrlGuard

        guard = UrlGuard(SecurityConfig(cdn_allowlist=sorted(self._allowlist())))
        assert guard.permits(
            "https://tracking.swap.fm/track/abc/mgln.ai/e/48/dts.podtrac.com/x.mp3",
            related_to="https://publicfeeds.net/f/8376/clickhere",
        )

    def test_an_unrelated_host_is_still_refused(self) -> None:
        """Widening the list must not have widened it to everything."""
        from podcast_agent.config import SecurityConfig
        from podcast_agent.net import UrlGuard

        guard = UrlGuard(SecurityConfig(cdn_allowlist=sorted(self._allowlist())))
        assert not guard.permits(
            "https://evil.example/x.mp3",
            related_to="https://publicfeeds.net/f/8376/clickhere",
        )


class TestModelEndpointsAreConfinedToTheFile:
    """`api_base` is the most consequential value in the configuration.

    Whoever can set it receives every prompt and every transcript the pipeline
    produces. The console may change which model runs; it may not invent a new
    place to send the work.
    """

    def _settings(self, tmp_path: Path, api_base: str | None = None) -> Settings:
        primary: dict[str, Any] = {"provider": "ollama", "model": "m"}
        if api_base:
            primary["api_base"] = api_base
        return make_settings(
            tmp_path,
            llm={
                "tiers": {
                    "tier0": {"primary": primary, "timeout_s": 30},
                    "tier1": {"primary": {"provider": "ollama", "model": "n"}, "timeout_s": 60},
                }
            },
        )

    def test_loopback_is_always_allowed(self, tmp_path: Path) -> None:
        allowed = allowed_api_base_hosts(self._settings(tmp_path))
        assert {"localhost", "127.0.0.1"} <= allowed

    def test_a_provider_reaches_its_own_endpoint(self, tmp_path: Path) -> None:
        assert "openrouter.ai" in allowed_api_base_hosts(self._settings(tmp_path))

    def test_a_host_named_in_the_file_is_allowed(self, tmp_path: Path) -> None:
        allowed = allowed_api_base_hosts(self._settings(tmp_path, "http://gpu-box.lan:11434"))
        assert "gpu-box.lan" in allowed

    def test_an_unknown_host_is_rejected(self, tmp_path: Path) -> None:
        baseline = self._settings(tmp_path)
        candidate = self._settings(tmp_path, "http://elsewhere.example:11434")
        with pytest.raises(OverrideRejected, match=r"elsewhere\.example"):
            check_api_bases(candidate, allowed_api_base_hosts(baseline))

    def test_the_rejection_says_how_to_authorise_it(self, tmp_path: Path) -> None:
        baseline = self._settings(tmp_path)
        candidate = self._settings(tmp_path, "http://elsewhere.example:11434")
        with pytest.raises(OverrideRejected, match=r"config\.yaml"):
            check_api_bases(candidate, allowed_api_base_hosts(baseline))

    def test_an_endpoint_without_a_base_is_not_examined(self, tmp_path: Path) -> None:
        """A provider that needs no `api_base` reaches its own endpoint."""
        baseline = self._settings(tmp_path)
        check_api_bases(baseline, allowed_api_base_hosts(baseline))

    def test_a_fallback_is_checked_too(self, tmp_path: Path) -> None:
        """The chain is walked in full — a fallback sends the same content."""
        baseline = self._settings(tmp_path)
        candidate = make_settings(
            tmp_path,
            llm={
                "tiers": {
                    "tier0": {
                        "primary": {"provider": "ollama", "model": "m"},
                        "fallbacks": [
                            {
                                "provider": "ollama",
                                "model": "m2",
                                "api_base": "http://elsewhere.example:11434",
                            }
                        ],
                        "timeout_s": 30,
                    },
                    "tier1": {"primary": {"provider": "ollama", "model": "n"}, "timeout_s": 60},
                }
            },
        )
        with pytest.raises(OverrideRejected, match=r"elsewhere\.example"):
            check_api_bases(candidate, allowed_api_base_hosts(baseline))

    def test_an_override_cannot_authorise_itself(self, tmp_path: Path) -> None:
        """The reason the check is judged against the file rather than the
        running configuration: once applied, a bad override would otherwise
        appear in the baseline and be permitted from then on."""
        already_running = self._settings(tmp_path, "http://elsewhere.example:11434")
        from_the_file = self._settings(tmp_path)
        assert "elsewhere.example" in allowed_api_base_hosts(already_running)
        assert "elsewhere.example" not in allowed_api_base_hosts(from_the_file)
