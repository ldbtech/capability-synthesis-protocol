"""
csp.orchestrator.credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CredentialStore  — persists API keys/tokens to disk, injects them as env vars
                   into the sandbox for synthesized capabilities that need them.

CredentialRequired — raised (or yielded as an event) when a synthesized
                     capability needs a credential that isn't in the store yet.

The store lives at  <credentials_dir>/credentials.json  (default: credentials/).
Each entry is keyed by the env-var name the synthesized code expects:

    {
      "OPENWEATHER_API_KEY": "abc123",
      "GITHUB_TOKEN":        "ghp_..."
    }

The synthesizer declares requirements via a  ##CREDENTIALS  block:

    ##CREDENTIALS
    OPENWEATHER_API_KEY: OpenWeatherMap · get at https://openweathermap.org/api
    GITHUB_TOKEN: GitHub personal token · get at https://github.com/settings/tokens

CSP parses those into CredentialSpec dicts and checks the store before
executing any synthesized capability that declares credentials.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("csp.credentials")


@dataclass
class CredentialSpec:
    """What a synthesized capability needs from the outside world."""
    env_key:    str             # e.g. "OPENWEATHER_API_KEY"
    service:    str             # e.g. "OpenWeatherMap"
    get_it_at:  str = ""        # URL where the user can obtain the key
    description: str = ""       # short description of what the key is for


class CredentialRequired(Exception):
    """
    Raised when a synthesized capability needs a credential that is not
    yet in the store.  The orchestrator catches this and emits a
    ``credential_required`` SSE event to the client.
    """
    def __init__(self, specs: list[CredentialSpec]) -> None:
        self.specs = specs
        super().__init__(f"missing credentials: {[s.env_key for s in specs]}")


class CredentialStore:
    """
    Persists API keys to  <root>/credentials.json  and exposes them as
    environment-variable dicts for sandbox injection.

    Parameters
    ----------
    root:
        Directory that holds credentials.json.  Created automatically.
    """

    def __init__(self, root: str = "credentials") -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._file = self._root / "credentials.json"
        self._data: dict[str, str] = self._load()
        self._write_gitignore()
        log.info("credential store ready at %s", self._root)

    # ── public API ────────────────────────────────────────────────────

    def has(self, env_key: str) -> bool:
        return env_key in self._data

    def get(self, env_key: str) -> Optional[str]:
        return self._data.get(env_key)

    def set(self, env_key: str, value: str) -> None:
        """Store a credential and flush to disk immediately."""
        self._data[env_key] = value
        self._flush()
        log.info("stored credential %r", env_key)

    def as_env(self) -> dict[str, str]:
        """Return all stored credentials as a flat env-var dict for sandbox injection."""
        return dict(self._data)

    def missing(self, specs: list[CredentialSpec]) -> list[CredentialSpec]:
        """Return only the specs whose env_key is not yet in the store."""
        return [s for s in specs if not self.has(s.env_key)]

    # ── internal ──────────────────────────────────────────────────────

    def _load(self) -> dict[str, str]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("could not load credentials file: %s", exc)
        return {}

    def _flush(self) -> None:
        try:
            self._file.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("could not persist credentials: %s", exc)

    def _write_gitignore(self) -> None:
        gi = self._root / ".gitignore"
        if not gi.exists():
            gi.write_text("# CSP credential store — never commit this\n*\n", encoding="utf-8")
