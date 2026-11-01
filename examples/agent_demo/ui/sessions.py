"""Signed-in visitors of the hosted demo, kept in memory only.

A session holds the visitor's gateway key so their runs can use it. The
key is never written to disk or logs: it lives here, and in the
environment of that visitor's own agent processes while they run. The
browser holds only a random session id, in an HttpOnly cookie. Sessions
expire after a while idle, and on sign-out; the instance restarting signs
everyone out (by design: nothing to leak from storage).
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

COOKIE = "tabctx_demo_session"


@dataclass
class Session:
    id: str
    key: str = field(repr=False)
    agents: dict[str, dict[str, str] | None]
    created: float = field(default_factory=time.time)
    seen: float = field(default_factory=time.time)

    @property
    def key_id(self) -> str:
        """The key's public id (vp_xxxx before the dot), for display."""
        return self.key.split(".", 1)[0]

    @property
    def slug(self) -> str:
        """A folder name for this visitor's runs (not the session id)."""
        return hashlib.sha256(self.id.encode()).hexdigest()[:16]


class SessionStore:
    def __init__(
        self,
        runs_root: Path,
        idle_seconds: float = 4 * 3600,
        on_end: Callable[[Path], None] | None = None,
    ) -> None:
        """on_end(runs folder): called before a session's runs are deleted
        (sign-out or expiry), to stop any agent still running there."""
        self.runs_root = runs_root
        self.idle_seconds = idle_seconds
        self.on_end = on_end
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, key: str, agents: dict[str, dict[str, str] | None]) -> Session:
        session = Session(secrets.token_urlsafe(32), key, agents)
        with self._lock:
            self._purge()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            self._purge()
            session = self._sessions.get(session_id)
            if session:
                session.seen = time.time()
            return session

    def end(self, session_id: str | None) -> None:
        with self._lock:
            session = self._sessions.pop(session_id or "", None)
        if session:
            self._remove_runs(session)

    def runs_dir(self, session: Session) -> Path:
        return self.runs_root / session.slug

    def _purge(self) -> None:
        cutoff = time.time() - self.idle_seconds
        for sid in [s for s, v in self._sessions.items() if v.seen < cutoff]:
            self._remove_runs(self._sessions.pop(sid))

    def _remove_runs(self, session: Session) -> None:
        folder = self.runs_dir(session)
        if self.on_end:
            self.on_end(folder)
        shutil.rmtree(folder, ignore_errors=True)
