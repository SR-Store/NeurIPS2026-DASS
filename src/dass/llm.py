"""OpenAI-compatible chat client for a configured model endpoint.

Standard library only, thread-safe; callers get concurrency from a thread pool.
Every call is counted; ``usage()`` gives the totals a run reports as its LLM cost.

Environment proxies are ignored unless ``env_proxy=True`` so local inference
endpoints can be contacted directly.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Reply:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    headers: dict


class LLMClient:
    def __init__(self, base_urls: list[str], model: str, sampling: dict | None = None,
                 api_key: str | None = None, timeout_s: float = 900, retries: int = 4,
                 env_proxy: bool = False):
        if not base_urls:
            raise ValueError("no LLM endpoint")
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.model = model
        self.sampling = dict(sampling or {})
        self._key = api_key
        self.timeout_s = timeout_s
        self.retries = retries
        self._opener = (urllib.request.build_opener() if env_proxy
                        else urllib.request.build_opener(urllib.request.ProxyHandler({})))
        self._next = itertools.cycle(range(len(self.base_urls)))
        self._lock = threading.Lock()
        self.calls = self.errors = self.prompt_tokens = self.completion_tokens = 0
        self.wall_s = 0.0

    def chat(self, messages: list[dict], **overrides) -> Reply:
        body = {"model": self.model, "messages": messages, **self.sampling, **overrides}
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        last = None
        for attempt in range(self.retries + 1):
            with self._lock:
                url = self.base_urls[next(self._next)] + "/v1/chat/completions"
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with self._opener.open(req, timeout=self.timeout_s) as r:
                    payload = json.loads(r.read())
                    resp_headers = dict(r.headers.items())
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                code = getattr(e, "code", None)
                with self._lock:
                    self.errors += 1
                if code is not None and 400 <= code < 500 and code != 429:
                    break
                time.sleep(min(60, 2 ** attempt))
                continue
            usage = payload.get("usage") or {}
            choice = payload["choices"][0]
            reply = Reply(
                text=choice["message"].get("content") or "",
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                headers=resp_headers,
            )
            with self._lock:
                self.calls += 1
                self.prompt_tokens += reply.prompt_tokens
                self.completion_tokens += reply.completion_tokens
                self.wall_s += time.time() - t0
            return reply
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {last!r}")

    def usage(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "errors": self.errors, "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens, "call_wall_s": round(self.wall_s, 2)}
