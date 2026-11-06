"""Opt-in HTTP with persisted budgets, retries and byte-exact raw responses."""

import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scripts.pivot_sources.common import SourceError, digest, now, read_json, safe_path, sha256, write_json


class BudgetExceeded(SourceError):
    pass


def validate_url(url, hosts):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in hosts or parts.username or parts.password or parts.port not in (None, 443):
        raise SourceError("Only HTTPS on the configured official hosts is allowed")
    if any(key.lower() in {"token", "key", "api_key", "apikey", "api-key", "password", "secret", "auth", "access_token", "authorization", "signature"}
           or key.lower().startswith("x-amz-") for key, _ in parse_qsl(parts.query)):
        raise SourceError("Do not put credentials or signed URLs in probe plans/logs")
    return url


class SafeRedirect(HTTPRedirectHandler):
    def __init__(self, hosts):
        super().__init__()
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, self.hosts)
        # Redirects are new requests. Never let urllib issue them outside budgets.
        raise SourceError(f"Redirect not followed; review the official destination explicitly: {newurl}")


class BoundedHTTP:
    def __init__(self, output, hosts, limits, allow_network=False, opener=None, sleeper=time.sleep, clock=time.monotonic):
        self.output, self.hosts, self.limits = Path(output), set(hosts), dict(limits)
        self.allow_network, self.sleeper, self.clock = allow_network, sleeper, clock
        if any(self.limits[key] <= 0 for key in ("max_requests", "max_file_bytes", "max_total_bytes", "timeout_seconds")):
            raise SourceError("HTTP limits must be positive")
        if self.limits["retries"] < 0 or self.limits["min_interval_seconds"] < 0:
            raise SourceError("Retry/rate values cannot be negative")
        self.opener = opener
        self.last_call = None
        self.journal_path = self.output / "http_manifest.json"
        self.state = read_json(self.journal_path) if self.journal_path.exists() else {
            "requests_used": 0, "bytes_received": 0, "responses": {}, "attempts": []}
        self.state.setdefault("bytes_budget_charged", self.state["bytes_received"])
        if self.state.get("pending_read_reservation", 0):
            # An abrupt termination may happen after socket.read but before its
            # count is saved. Charge the reserved bytes conservatively, without
            # inventing how many bytes were actually received.
            reserved = self.state["pending_read_reservation"]
            self.state["bytes_budget_charged"] += reserved
            self.state["unverified_interrupted_read_reservations"] = self.state.get("unverified_interrupted_read_reservations", 0) + reserved
            self.state["pending_read_reservation"] = 0
            self.checkpoint()

    def checkpoint(self):
        # This run-local journal is the sole mutable download metadata. Raw
        # attempts and sealed bundles are never overwritten.
        temporary = self.output / "http_manifest.json.tmp"
        write_json(temporary, self.state, exclusive=False)
        temporary.replace(self.journal_path)

    def fetch(self, url, method="GET", allow_binary=False):
        if not self.allow_network:
            raise SourceError("Network disabled; pass --allow-network explicitly")
        validate_url(url, self.hosts)
        if method not in ("GET", "HEAD"):
            raise SourceError("Only GET/HEAD are allowed")
        fingerprint = digest([method, url, allow_binary])
        prior = self.state["responses"].get(fingerprint)
        if prior:
            path = safe_path(self.output, prior["raw_file"])
            if sha256(path) != prior["sha256"]:
                raise SourceError("Cached raw response changed")
            return path.read_bytes(), prior
        if self.opener is None:
            self.opener = build_opener(SafeRedirect(self.hosts))
        for retry in range(self.limits["retries"] + 1):
            if self.state["requests_used"] >= self.limits["max_requests"]:
                raise BudgetExceeded("Maximum HTTP attempts reached, including retries")
            remaining = self.limits["max_total_bytes"] - self.state["bytes_budget_charged"]
            if remaining <= 0:
                raise BudgetExceeded("Maximum total bytes reached")
            if self.last_call is not None:
                self.sleeper(max(0, self.limits["min_interval_seconds"] - (self.clock() - self.last_call)))
            attempt_id = self.state["requests_used"] + 1
            self.state["requests_used"] = attempt_id
            self.last_call = self.clock()
            entry = {"url": url, "method": method, "attempt": attempt_id, "requested_at": now(), "status": "pending"}
            self.state["attempts"].append(entry)
            self.checkpoint()
            request = Request(url, method=method, headers={"User-Agent": "PIVOT-Stage1.6-SourceProbe/1.0 (manual bounded research probe)",
                                                          "Accept-Encoding": "identity"})
            response = None
            raw_path = None
            try:
                response = self.opener.open(request, timeout=self.limits["timeout_seconds"])
                status = response.status
                if not 200 <= status < 300:
                    raise SourceError(f"HTTP status {status}")
                headers = response.headers
                declared = headers.get("Content-Length")
                cap = min(self.limits["max_file_bytes"], remaining)
                if method != "HEAD" and declared and int(declared) > cap:
                    raise BudgetExceeded("Content-Length exceeds per-file/remaining byte budget; body not downloaded")
                if declared and int(declared) < 0:
                    raise SourceError("Invalid negative Content-Length")
                if method != "HEAD" and declared:
                    cap = min(cap, int(declared))
                if method != "HEAD" and not allow_binary and any(token in headers.get("Content-Type", "").lower()
                                                                 for token in ("grib", "octet-stream", "zip", "gzip")):
                    raise SourceError("Binary/global payload refused; use metadata or explicitly reviewed small-sample mode")
                raw_dir = self.output / "raw"
                raw_dir.mkdir(exist_ok=True)
                raw_path = raw_dir / f"{fingerprint}_{attempt_id:05d}.raw"
                entry["raw_file"] = raw_path.relative_to(self.output).as_posix()
                received = 0
                with raw_path.open("xb") as handle:
                    while method != "HEAD":
                        # Never ask a server for more than the remaining cap.
                        size = min(65536, cap - received)
                        if size <= 0:
                            if declared and int(declared) == received:
                                break
                            raise BudgetExceeded("Undeclared/oversized response reached byte cap; partial raw kept, not marked successful")
                        self.state["pending_read_reservation"] = size
                        self.checkpoint()
                        chunk = response.read(size)
                        self.state["pending_read_reservation"] = 0
                        if not chunk:
                            break
                        received += len(chunk)
                        self.state["bytes_received"] += len(chunk)
                        self.state["bytes_budget_charged"] += len(chunk)
                        handle.write(chunk)
                        self.checkpoint()
                if method != "HEAD" and declared and received != int(declared):
                    raise SourceError("Truncated response; Content-Length mismatch")
                entry.update(status="complete", http_status=status, received_at_now=now(),
                             bytes=received, advertised_bytes=int(declared) if declared else None,
                             raw_file=raw_path.relative_to(self.output).as_posix(), sha256=sha256(raw_path),
                             headers={key: headers.get(key) for key in ("Content-Type", "Content-Length", "Last-Modified", "ETag")},
                             historical_received_at=None,
                             clock_note="request retrieval time and HTTP Last-Modified are NOT verified historical issuance/receipt")
                self.state["responses"][fingerprint] = dict(entry)
                self.checkpoint()
                return raw_path.read_bytes(), entry
            except (HTTPError, URLError, TimeoutError, OSError, SourceError, ValueError) as exc:
                if self.state.get("pending_read_reservation", 0):
                    reserved = self.state["pending_read_reservation"]
                    self.state["bytes_budget_charged"] += reserved
                    self.state["unverified_interrupted_read_reservations"] = self.state.get("unverified_interrupted_read_reservations", 0) + reserved
                    self.state["pending_read_reservation"] = 0
                entry.update(status="failed", error_type=type(exc).__name__,
                             error=f"HTTP {exc.code}" if isinstance(exc, HTTPError) else str(exc))
                if raw_path is not None and raw_path.is_file():
                    entry.update(partial_raw_sha256=sha256(raw_path), partial_raw_bytes=raw_path.stat().st_size,
                                 raw_payload_complete=False)
                self.checkpoint()
                retryable = isinstance(exc, (URLError, TimeoutError, OSError)) and not isinstance(exc, HTTPError)
                retryable = retryable or isinstance(exc, HTTPError) and (exc.code == 429 or 500 <= exc.code < 600)
                if retry == self.limits["retries"] or not retryable:
                    raise SourceError(entry["error"]) from exc
                self.sleeper(min(30, 2 ** retry))
            finally:
                if response is not None:
                    response.close()
        raise SourceError("HTTP request did not complete")
