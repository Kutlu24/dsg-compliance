"""Guard for /ask: every question costs an embedding search and an LLM call on a free-tier key behind a public page.

Per-client and global rate limits, and request bodies are capped while they stream in (not only by their declared
length). One pure-ASGI layer in front of the app. The client is the right-most X-Forwarded-For entry (the one our own
proxy appended; anything further left is whatever the caller typed), IPv6 clients are counted per /56, and the table of
clients has a fixed size (plain LRU).
"""
# Known trade-off: the global budget is what protects the free-tier key, and someone with many addresses can use it up
# for one window (10 minutes), during which everyone gets 429. Without it the same person could spend the key's whole
# daily quota, which is the longer outage. Only requests that pass validation are charged.
from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
from collections import OrderedDict, deque

WINDOW_S = 600.0
MAX_CLIENTS = 5000
# path ending -> (requests per client per window, requests in total per window)
RULES: dict[str, tuple[int, int]] = {
    "/ask": (15, 200),
}
DEFAULT_BODY = 32 * 1024
BODY_LIMITS: dict[str, int] = {}


def rule_for(path: str, table: dict):
    return next((k for k in table if path == k or path.endswith(k)), None)


def bucket(addr: str) -> str:
    try:
        ip = ipaddress.ip_address(addr.strip())
    except ValueError:
        return "unknown"
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/56", strict=False).network_address) + "/56"  # what a home or small site is usually given
    return str(ip)

try:
    PROXY_HOPS = max(1, int(os.environ.get("PROXY_HOPS", "1")))
except ValueError:
    PROXY_HOPS = 1


# loopback, the private IPv4 ranges, carrier-grade NAT (Tailscale) and IPv6 unique-local: where a proxy of ours can sit.
# Spelled out because ipaddress.is_private also covers documentation and other special ranges.
_PROXY_NETS = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
                                                  "100.64.0.0/10", "::1/128", "fc00::/7")]


def peer_is_proxy(peer: str) -> bool:
    """Is the machine that connected to us a reverse proxy of ours (Render's router, Caddy on the same host or Docker
    network)? Those sit on loopback or a private network. A caller that reaches the app directly from the internet has
    a public address, and then nothing it writes into X-Forwarded-For is believed."""
    try:
        ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip in net for net in _PROXY_NETS)


def client_of(peer: str, xff: list[str]) -> str:
    """The address to count. Behind our own proxy: the X-Forwarded-For entry PROXY_HOPS from the right (each proxy
    appends the address it saw, so entries further left are whatever the caller typed). Otherwise: the peer itself."""
    if xff and peer_is_proxy(peer):
        return bucket(xff[-min(PROXY_HOPS, len(xff))])
    return bucket(peer or "unknown")


class Limiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._global: dict[str, deque[float]] = {}

    def check(self, path: str, client: str) -> int:
        """0 = allowed (and counted); otherwise the seconds to wait."""
        per_client, total = RULES[path]
        now = self._clock()
        with self._lock:
            g = self._global.setdefault(path, deque())
            while g and now - g[0] > WINDOW_S:
                g.popleft()
            if len(g) >= total:
                return max(1, int(g[0] + WINDOW_S - now) + 1)
            key = (path, client)
            if key not in self._clients and len(self._clients) >= MAX_CLIENTS:
                self._clients.popitem(last=False)
            q = self._clients.setdefault(key, deque())
            self._clients.move_to_end(key)
            while q and now - q[0] > WINDOW_S:
                q.popleft()
            if len(q) >= per_client:
                return max(1, int(q[0] + WINDOW_S - now) + 1)
            q.append(now)
            g.append(now)
            return 0

    def refund(self, path: str, client: str) -> None:
        """Take back the most recent hit: the request was refused by validation, so nothing costly ran."""
        with self._lock:
            for q in (self._clients.get((path, client)), self._global.get(path)):
                if q:
                    q.pop()

    def reset(self) -> None:
        with self._lock:
            self._clients.clear()
            self._global.clear()


limiter = Limiter()


class _TooLarge(Exception):
    pass


class Guard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        xff, lengths = [], []
        for k, v in scope.get("headers", []):
            if k.lower() == b"x-forwarded-for":
                xff += [p.strip() for p in v.decode("latin-1").split(",") if p.strip()]
            elif k.lower() == b"content-length":
                lengths.append(v.decode("latin-1").strip())

        async def reply(status: int, detail: str, extra: list | None = None) -> None:
            body = json.dumps({"detail": detail}).encode()
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                    (b"x-content-type-options", b"nosniff")] + (extra or [])})
            await send({"type": "http.response.body", "body": body})

        limit = BODY_LIMITS.get(rule_for(path, BODY_LIMITS), DEFAULT_BODY)
        if len(lengths) > 1 or any(not n.isdigit() for n in lengths):
            await reply(400, "Bad Content-Length.")
            return
        if lengths and int(lengths[0]) > limit:
            await reply(413, "Request too large.")
            return
        charged = None
        rule = rule_for(path, RULES)
        if rule is not None:
            client = client_of((scope.get("client") or [""])[0] or "", xff)
            wait = limiter.check(rule, client)
            charged = (rule, client)
            if wait:
                await reply(429, "Too many requests in a short time. Please wait a few minutes.", [(b"retry-after", str(wait).encode())])
                return
        state = {"n": 0, "over": False, "started": False}

        async def counting():
            msg = await receive()
            if msg["type"] == "http.request":
                state["n"] += len(msg.get("body", b""))
                if state["n"] > limit:
                    state["over"] = True
                    raise _TooLarge
            return msg

        async def guarded(msg):
            if state["over"]:
                return
            if msg["type"] == "http.response.start":
                state["started"] = True
                if charged and 400 <= msg["status"] < 500:  # refused by validation: no model or solver ran, so it does not count
                    limiter.refund(*charged)
            await send(msg)

        try:
            await self.app(scope, counting, guarded)
        except BaseException:  # noqa: BLE001 - re-raised unless it is our own size abort
            if not state["over"]:
                raise
        if state["over"] and not state["started"]:
            await reply(413, "Request too large.")
