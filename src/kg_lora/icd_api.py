"""Small, defensive client for the WHO ICD-11 MMS API.

The API returns ``http://`` entity identifiers even though WHO recommends that
clients call their ``https://`` equivalents.  This module therefore upgrades
those identifiers before making the second (entity-title) request and rejects
non-TLS configuration outright.

Only access tokens are cached in memory.  API response caches contain public
ICD data and request metadata, never credentials or request headers.
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)


DEFAULT_BASE_URL = "https://id.who.int"
DEFAULT_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
SENSITIVE_KEY_RE = re.compile(
    r"(?i)(access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|"
    r"authorization|password|secret)"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
JSON_SECRET_RE = re.compile(
    r'(?i)("(?:access[_-]?token|refresh[_-]?token|id[_-]?token|'
    r'client[_-]?secret|authorization|password|secret)"\s*:\s*)"[^"]*"'
)
CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._&/+*:-]{0,255}$")


class ICDAPIConfigurationError(ValueError):
    """Raised when required configuration is missing or unsafe."""


class ICDAPIError(RuntimeError):
    """A redacted WHO request or response error safe for logs and reports."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status: int | None = None,
        response_summary: str = "",
        sensitive_values: tuple[str, ...] = (),
    ) -> None:
        self.endpoint = _safe_endpoint(endpoint)
        self.status = status
        self.response_summary = _safe_summary(response_summary, sensitive_values)
        status_text = str(status) if status is not None else "unavailable"
        detail = (
            f"WHO ICD API {message}; endpoint={self.endpoint}; "
            f"HTTP status={status_text}"
        )
        if self.response_summary:
            detail += f"; response={self.response_summary}"
        super().__init__(detail)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic record without credentials."""

        return {
            "message": str(self),
            "endpoint": self.endpoint,
            "http_status": self.status,
            "response_summary": self.response_summary,
        }


class _NetworkError(OSError):
    """Internal transport error with no request headers attached."""


class _HTTPResponse:
    def __init__(
        self,
        status_code: int,
        body: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = body.decode("utf-8", errors="replace")
        self.headers = dict(headers or {})

    def json(self) -> Any:
        return json.loads(self.text)


class _HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        if urlparse(newurl).scheme.lower() != "https":
            raise URLError("refused a non-HTTPS redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _UrllibSession:
    """Minimal requests-shaped HTTPS transport backed by the standard library."""

    def __init__(self, context: ssl.SSLContext) -> None:
        self._opener = build_opener(
            HTTPSHandler(context=context), _HTTPSOnlyRedirectHandler()
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float,
        verify: bool,
        allow_redirects: bool,
    ) -> _HTTPResponse:
        del allow_redirects  # The installed handler follows HTTPS-only redirects.
        if not verify:
            raise _NetworkError("TLS certificate verification cannot be disabled")
        request_headers = dict(headers or {})
        if auth:
            credentials = f"{auth[0]}:{auth[1]}".encode("utf-8")
            request_headers["Authorization"] = (
                "Basic " + base64.b64encode(credentials).decode("ascii")
            )
        encoded_data = None
        if data is not None:
            encoded_data = urlencode(data).encode("utf-8")
            request_headers.setdefault(
                "Content-Type", "application/x-www-form-urlencoded"
            )
        request = Request(
            url, data=encoded_data, headers=request_headers, method=method.upper()
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                final_url = response.geturl()
                if urlparse(final_url).scheme.lower() != "https":
                    raise _NetworkError("request ended on a non-HTTPS endpoint")
                return _HTTPResponse(
                    int(response.status), response.read(), dict(response.headers)
                )
        except HTTPError as exc:
            return _HTTPResponse(int(exc.code), exc.read(), dict(exc.headers or {}))
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise _NetworkError(f"{type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class ICDAPIConfig:
    """Runtime settings for a pinned ICD-11 MMS release."""

    client_id: str
    client_secret: str = field(repr=False)
    release: str = "2025-01"
    language: str = "en"
    api_version: str = "v2"
    base_url: str = DEFAULT_BASE_URL
    token_url: str = DEFAULT_TOKEN_URL
    cache_dir: Path = Path(".cache/icd-api/2025-01/en")
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ICDAPIConfigurationError("WHO_ICD_CLIENT_ID is required")
        if not self.client_secret:
            raise ICDAPIConfigurationError("WHO_ICD_CLIENT_SECRET is required")
        if self.api_version != "v2":
            raise ICDAPIConfigurationError("WHO_ICD_API_VERSION must be v2")
        if not re.fullmatch(r"\d{4}-\d{2}", self.release):
            raise ICDAPIConfigurationError(
                "WHO_ICD_RELEASE must use the YYYY-MM release form"
            )
        if not re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z0-9]{2,8})*", self.language):
            raise ICDAPIConfigurationError(
                "WHO_ICD_LANGUAGE must be an ISO language tag such as en"
            )
        _require_https(self.base_url, "WHO_ICD_BASE_URL")
        _require_https(self.token_url, "WHO_ICD_TOKEN_URL")
        if self.timeout_seconds <= 0:
            raise ICDAPIConfigurationError("WHO_ICD_TIMEOUT_SECONDS must be positive")
        if self.max_retries < 0:
            raise ICDAPIConfigurationError("WHO_ICD_MAX_RETRIES cannot be negative")
        if self.retry_backoff_seconds < 0:
            raise ICDAPIConfigurationError(
                "WHO_ICD_RETRY_BACKOFF_SECONDS cannot be negative"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ICDAPIConfig":
        """Build configuration from already-loaded environment variables."""

        env = os.environ if environ is None else environ
        release = env.get("WHO_ICD_RELEASE", "2025-01").strip()
        language = env.get("WHO_ICD_LANGUAGE", "en").strip()
        default_cache = Path(".cache") / "icd-api" / release / language
        cache_value = env.get("ICD_API_CACHE_DIR") or env.get("WHO_ICD_CACHE_DIR")
        try:
            timeout_seconds = float(env.get("WHO_ICD_TIMEOUT_SECONDS", "30"))
            max_retries = int(env.get("WHO_ICD_MAX_RETRIES", "3"))
            retry_backoff = float(
                env.get("WHO_ICD_RETRY_BACKOFF_SECONDS", "0.5")
            )
        except ValueError as exc:
            raise ICDAPIConfigurationError(
                "WHO ICD timeout/retry environment values must be numeric"
            ) from exc
        return cls(
            client_id=env.get("WHO_ICD_CLIENT_ID", ""),
            client_secret=env.get("WHO_ICD_CLIENT_SECRET", ""),
            release=release,
            language=language,
            api_version=env.get("WHO_ICD_API_VERSION", "v2").strip(),
            base_url=env.get("WHO_ICD_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            token_url=env.get("WHO_ICD_TOKEN_URL", DEFAULT_TOKEN_URL),
            cache_dir=Path(cache_value).expanduser() if cache_value else default_cache,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff,
        )


@dataclass(frozen=True)
class ICDCodeResult:
    """The joined result of MMS ``codeinfo`` and entity requests."""

    requested_code: str
    canonical_code: str
    stem_code: str
    entity_uri: str
    title: str
    release: str
    language: str
    codeinfo_endpoint: str
    entity_endpoint: str
    codeinfo: dict[str, Any] = field(repr=False)
    entity: dict[str, Any] = field(repr=False)

    def as_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_code": self.requested_code,
            "canonical_code": self.canonical_code,
            "stem_code": self.stem_code,
            "entity_uri": self.entity_uri,
            "title": self.title,
            "release": self.release,
            "language": self.language,
            "codeinfo_endpoint": self.codeinfo_endpoint,
            "entity_endpoint": self.entity_endpoint,
        }
        if include_payloads:
            payload["codeinfo"] = self.codeinfo
            payload["entity"] = self.entity
        return payload


@dataclass
class _TokenEntry:
    access_token: str = field(repr=False)
    usable_until: float


_TOKEN_CACHE: dict[tuple[str, str], _TokenEntry] = {}
_TOKEN_LOCK = threading.RLock()
_CACHE_WRITE_LOCK = threading.Lock()


def _clear_process_token_cache_for_tests() -> None:
    """Reset process state.  Intended only for isolated unit tests."""

    with _TOKEN_LOCK:
        _TOKEN_CACHE.clear()


def _require_https(url: str, variable_name: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ICDAPIConfigurationError(f"{variable_name} must be an https URL")


def _safe_endpoint(endpoint: str) -> str:
    """Remove credential-like query parameters from a URL before reporting it."""

    try:
        parsed = urlparse(endpoint)
        safe_query = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            safe_query.append((key, "[REDACTED]" if SENSITIVE_KEY_RE.search(key) else value))
        return urlunparse(parsed._replace(query=urlencode(safe_query)))
    except Exception:
        return "[invalid endpoint]"


def _safe_summary(
    value: Any,
    sensitive_values: tuple[str, ...] = (),
    *,
    limit: int = 600,
) -> str:
    text = " ".join(str(value or "").split())
    text = JSON_SECRET_RE.sub(r'\1"[REDACTED]"', text)
    text = BEARER_RE.sub("Bearer [REDACTED]", text)
    for sensitive in sensitive_values:
        if sensitive and len(sensitive) >= 3:
            text = text.replace(sensitive, "[REDACTED]")
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


def _localized_text(value: Any, language: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("@value", "value", "label", "title"):
            if isinstance(value.get(key), str):
                return value[key].strip()
        return ""
    if isinstance(value, list):
        candidates = [item for item in value if isinstance(item, (str, dict))]
        for item in candidates:
            if isinstance(item, dict) and item.get("@language") == language:
                return _localized_text(item, language)
        return _localized_text(candidates[0], language) if candidates else ""
    return ""


class WHOICDClient:
    """OAuth2 client for a single ICD-11 MMS release and language."""

    def __init__(
        self,
        config: ICDAPIConfig,
        *,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._monotonic = monotonic
        # Creating a default context is also a fail-fast check that the host has
        # a usable CA trust store. The transport receives verify=True explicitly.
        self._tls_context = ssl.create_default_context()
        self._session = session or _UrllibSession(self._tls_context)

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "WHOICDClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def _token_cache_key(self) -> tuple[str, str]:
        client_hash = hashlib.sha256(self.config.client_id.encode("utf-8")).hexdigest()
        return (self.config.token_url, client_hash)

    def _sensitive_values(self, *extra: str) -> tuple[str, ...]:
        values = [self.config.client_secret]
        values.extend(value for value in extra if value)
        return tuple(values)

    def _get_access_token(self) -> str:
        key = self._token_cache_key
        now = self._monotonic()
        with _TOKEN_LOCK:
            cached = _TOKEN_CACHE.get(key)
            if cached and now < cached.usable_until:
                return cached.access_token

            token_payload = self._request_json(
                "POST",
                self.config.token_url,
                authenticated=False,
                headers={"Accept": "application/json"},
                data={"grant_type": "client_credentials", "scope": "icdapi_access"},
                auth=(self.config.client_id, self.config.client_secret),
            )
            token = token_payload.get("access_token")
            if not isinstance(token, str) or not token:
                raise ICDAPIError(
                    "token response did not contain access_token",
                    endpoint=self.config.token_url,
                    status=200,
                    response_summary=json.dumps(token_payload, ensure_ascii=False),
                    sensitive_values=self._sensitive_values(),
                )
            try:
                expires_in = float(token_payload.get("expires_in", 3600))
            except (TypeError, ValueError):
                expires_in = 3600.0
            expires_in = max(1.0, expires_in)
            refresh_margin = min(60.0, max(1.0, expires_in * 0.1))
            usable_until = self._monotonic() + max(0.0, expires_in - refresh_margin)
            _TOKEN_CACHE[key] = _TokenEntry(token, usable_until)
            return token

    def _invalidate_access_token(self, token: str) -> None:
        with _TOKEN_LOCK:
            cached = _TOKEN_CACHE.get(self._token_cache_key)
            if cached and cached.access_token == token:
                _TOKEN_CACHE.pop(self._token_cache_key, None)

    def _retry_delay(self, attempt: int, response: Any | None = None) -> float:
        retry_after = ""
        if response is not None:
            retry_after = str(getattr(response, "headers", {}).get("Retry-After", ""))
        try:
            if retry_after:
                return min(30.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
        return min(30.0, self.config.retry_backoff_seconds * (2**attempt))

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        authenticated: bool,
        headers: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        _require_https(endpoint, "WHO ICD request endpoint")
        request_headers = dict(headers or {})
        access_token = ""
        if authenticated:
            access_token = self._get_access_token()
            request_headers.update(
                {
                    "Accept": "application/json",
                    "Accept-Language": self.config.language,
                    "API-Version": "v2",
                    "Authorization": f"Bearer {access_token}",
                }
            )

        max_attempts = self.config.max_retries + 1
        for attempt in range(max_attempts):
            try:
                response = self._session.request(
                    method,
                    endpoint,
                    headers=request_headers,
                    data=data,
                    auth=auth,
                    timeout=self.config.timeout_seconds,
                    verify=True,
                    allow_redirects=True,
                )
            except (TimeoutError, socket.timeout, _NetworkError) as exc:
                if attempt + 1 < max_attempts:
                    self._sleep(self._retry_delay(attempt))
                    continue
                raise ICDAPIError(
                    "network request failed",
                    endpoint=endpoint,
                    response_summary=f"{type(exc).__name__}: {exc}",
                    sensitive_values=self._sensitive_values(access_token),
                ) from exc
            except OSError as exc:
                raise ICDAPIError(
                    "request setup failed",
                    endpoint=endpoint,
                    response_summary=f"{type(exc).__name__}: {exc}",
                    sensitive_values=self._sensitive_values(access_token),
                ) from exc

            status = int(getattr(response, "status_code", 0))
            response_text = getattr(response, "text", "")
            if status in RETRYABLE_STATUS_CODES and attempt + 1 < max_attempts:
                self._sleep(self._retry_delay(attempt, response))
                continue
            if status < 200 or status >= 300:
                raise ICDAPIError(
                    "request failed",
                    endpoint=endpoint,
                    status=status,
                    response_summary=response_text,
                    sensitive_values=self._sensitive_values(access_token),
                )
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise ICDAPIError(
                    "response was not valid JSON",
                    endpoint=endpoint,
                    status=status,
                    response_summary=response_text,
                    sensitive_values=self._sensitive_values(access_token),
                ) from exc
            if not isinstance(payload, dict):
                raise ICDAPIError(
                    "response JSON was not an object",
                    endpoint=endpoint,
                    status=status,
                    response_summary=response_text,
                    sensitive_values=self._sensitive_values(access_token),
                )
            return payload
        raise AssertionError("request retry loop exhausted unexpectedly")

    def _authenticated_get_json(self, endpoint: str) -> dict[str, Any]:
        try:
            return self._request_json("GET", endpoint, authenticated=True)
        except ICDAPIError as exc:
            # Do not mistake a 401 from the token endpoint for an expired bearer
            # token. Only an authenticated resource endpoint gets one refresh.
            if exc.status != 401 or exc.endpoint != _safe_endpoint(endpoint):
                raise
            # A process-cached token may have been revoked before its advertised
            # expiry. Refresh it once, independent of transient retry settings.
            cached = _TOKEN_CACHE.get(self._token_cache_key)
            if cached:
                self._invalidate_access_token(cached.access_token)
            return self._request_json("GET", endpoint, authenticated=True)

    def _cache_path(self, kind: str, endpoint: str) -> Path:
        fingerprint_source = "\n".join(
            (endpoint, self.config.release, self.config.language, self.config.api_version)
        )
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        return self.config.cache_dir / f"{kind}-{fingerprint}.json"

    def _read_cache(self, path: Path, endpoint: str) -> dict[str, Any] | None:
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(wrapper, dict):
            return None
        if (
            wrapper.get("cache_version") != 1
            or wrapper.get("endpoint") != endpoint
            or wrapper.get("release") != self.config.release
            or wrapper.get("language") != self.config.language
            or wrapper.get("api_version") != self.config.api_version
            or not isinstance(wrapper.get("payload"), dict)
        ):
            return None
        return wrapper["payload"]

    def _write_cache(self, path: Path, endpoint: str, payload: dict[str, Any]) -> None:
        wrapper = {
            "cache_version": 1,
            "endpoint": endpoint,
            "release": self.config.release,
            "language": self.config.language,
            "api_version": self.config.api_version,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            serialized = json.dumps(
                wrapper, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            ) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with _CACHE_WRITE_LOCK:
                temporary.write_text(serialized, encoding="utf-8")
                os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ICDAPIError(
                "public response cache write failed",
                endpoint=endpoint,
                status=200,
                response_summary=f"{type(exc).__name__}; cache_path={path}",
                sensitive_values=self._sensitive_values(),
            ) from exc

    def _cached_get_json(
        self,
        endpoint: str,
        *,
        kind: str,
        force_refresh: bool,
    ) -> dict[str, Any]:
        path = self._cache_path(kind, endpoint)
        if not force_refresh:
            cached = self._read_cache(path, endpoint)
            if cached is not None:
                return cached
        payload = self._authenticated_get_json(endpoint)
        self._write_cache(path, endpoint, payload)
        return payload

    def _codeinfo_endpoint(self, code: str) -> str:
        encoded = quote(code, safe=".")
        return (
            f"{self.config.base_url}/icd/release/11/{self.config.release}"
            f"/mms/codeinfo/{encoded}"
        )

    def _entity_endpoint(self, entity_uri: str, codeinfo_endpoint: str) -> str:
        parsed = urlparse(entity_uri)
        base = urlparse(self.config.base_url)
        if not parsed.hostname or parsed.hostname.lower() != base.hostname.lower():
            raise ICDAPIError(
                "codeinfo returned an entity on an unexpected host",
                endpoint=codeinfo_endpoint,
                status=200,
                response_summary=f"entity_uri={_safe_endpoint(entity_uri)}",
                sensitive_values=self._sensitive_values(),
            )
        return urlunparse(parsed._replace(scheme="https", netloc=base.netloc))

    def lookup_code(self, code: str, *, force_refresh: bool = False) -> ICDCodeResult:
        """Validate an MMS code and fetch its localized entity title."""

        normalized_code = str(code).strip().upper()
        if not CODE_RE.fullmatch(normalized_code):
            raise ICDAPIError(
                "invalid ICD code syntax",
                endpoint=self.config.base_url,
                response_summary="code must be 1-256 characters without whitespace",
                sensitive_values=self._sensitive_values(),
            )

        codeinfo_endpoint = self._codeinfo_endpoint(normalized_code)
        codeinfo = self._cached_get_json(
            codeinfo_endpoint, kind="codeinfo", force_refresh=force_refresh
        )
        entity_uri = codeinfo.get("stemId")
        if not isinstance(entity_uri, str) or not entity_uri:
            raise ICDAPIError(
                "codeinfo response did not contain stemId",
                endpoint=codeinfo_endpoint,
                status=200,
                response_summary=json.dumps(codeinfo, ensure_ascii=False),
                sensitive_values=self._sensitive_values(),
            )
        entity_endpoint = self._entity_endpoint(entity_uri, codeinfo_endpoint)
        entity = self._cached_get_json(
            entity_endpoint, kind="entity", force_refresh=force_refresh
        )
        title = _localized_text(entity.get("title"), self.config.language)
        if not title:
            raise ICDAPIError(
                "entity response did not contain a localized title",
                endpoint=entity_endpoint,
                status=200,
                response_summary=json.dumps(entity, ensure_ascii=False),
                sensitive_values=self._sensitive_values(),
            )

        canonical_code = str(codeinfo.get("code") or normalized_code)
        stem_code = str(codeinfo.get("stemCode") or canonical_code)
        return ICDCodeResult(
            requested_code=normalized_code,
            canonical_code=canonical_code,
            stem_code=stem_code,
            entity_uri=entity_uri,
            title=title,
            release=self.config.release,
            language=self.config.language,
            codeinfo_endpoint=codeinfo_endpoint,
            entity_endpoint=entity_endpoint,
            codeinfo=codeinfo,
            entity=entity,
        )
