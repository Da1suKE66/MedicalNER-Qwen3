"""Small, defensive client for the WHO ICD-11 MMS API.

The API returns ``http://`` entity identifiers even though WHO recommends that
clients call their ``https://`` equivalents.  This module therefore upgrades
those identifiers before making the second (entity-title) request and rejects
non-TLS configuration outright.

Only access tokens are cached in memory. API response caches contain public ICD
candidate text and request metadata, never credentials or request headers. Raw
search/autocode parameters are represented by digests and explicit query-echo
fields are redacted, but an authoritative WHO title or matching label can equal
the query and therefore remain in the cached public candidate payload. Cache
files are written with owner-only permissions and must not be treated as
anonymized data.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import re
import socket
import ssl
import threading
import time
import unicodedata
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
ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PRIVATE_QUERY_KEYS = frozenset({"q", "searchtext"})
PRIVATE_RESPONSE_KEYS = frozenset(
    {"q", "query", "searchtext", "search_text", "words", "wordsuggestions"}
)
HTML_TAG_RE = re.compile(r"<[^>]*>")
MAX_SEARCH_QUERY_CHARACTERS = 512
MAX_AUTOCODE_TEXT_CHARACTERS = 1_000
MAX_DISCOVERY_GET_URL_BYTES = 2_000


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


def _expand_environment_references(
    value: str,
    environ: Mapping[str, str],
    *,
    setting_name: str,
) -> str:
    """Expand ``${NAME}`` using the supplied environment mapping.

    ``os.path.expandvars`` is intentionally not used because callers may pass a
    test or managed mapping that differs from the process environment. Missing
    and cyclic references fail closed instead of becoming literal directories.
    """

    expanded = value
    for _depth in range(20):
        references = ENV_REFERENCE_RE.findall(expanded)
        if not references:
            if not expanded.strip():
                raise ICDAPIConfigurationError(
                    f"{setting_name} expanded to an empty value"
                )
            return expanded
        missing = sorted({name for name in references if name not in environ})
        if missing:
            raise ICDAPIConfigurationError(
                f"{setting_name} references undefined environment variable "
                f"{missing[0]}"
            )
        updated = ENV_REFERENCE_RE.sub(
            lambda match: str(environ[match.group(1)]), expanded
        )
        if updated == expanded:
            break
        expanded = updated
    raise ICDAPIConfigurationError(
        f"{setting_name} contains cyclic or excessively nested references"
    )


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
        if cache_value:
            cache_value = _expand_environment_references(
                cache_value,
                env,
                setting_name="ICD_API_CACHE_DIR",
            )
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


@dataclass(frozen=True)
class ICDSearchCandidate:
    """One unverified MMS search/autocode candidate.

    Search and autocode are discovery endpoints, not code validation endpoints.
    Callers must require an unambiguous exact-title match and confirm ``code``
    with :meth:`WHOICDClient.lookup_code` before attaching it to an entity.
    """

    rank: int
    title: str
    code: str
    entity_uri: str
    foundation_uri: str
    matching_text: str
    match_score: float | None
    exact_title_match: bool
    retrieval_method: str
    used_flexisearch: bool
    endpoint: str
    payload: dict[str, Any] = field(repr=False)
    requires_code_validation: bool = field(default=True, init=False)

    def as_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rank": self.rank,
            "title": self.title,
            "code": self.code,
            "entity_uri": self.entity_uri,
            "foundation_uri": self.foundation_uri,
            "matching_text": self.matching_text,
            "match_score": self.match_score,
            "exact_title_match": self.exact_title_match,
            "retrieval_method": self.retrieval_method,
            "used_flexisearch": self.used_flexisearch,
            "endpoint": self.endpoint,
            "requires_code_validation": True,
        }
        if include_payload:
            result["payload"] = self.payload
        return result


@dataclass(frozen=True)
class ICDSearchResult:
    """Candidate-only result from MMS search or autocode discovery."""

    query: str
    retrieval_method: str
    used_flexisearch: bool
    attempted_endpoints: tuple[str, ...]
    candidates: tuple[ICDSearchCandidate, ...]
    payload: dict[str, Any] = field(repr=False)

    @property
    def exact_title_candidates(self) -> tuple[ICDSearchCandidate, ...]:
        """Return exact-title candidates without accepting any candidate."""

        return tuple(item for item in self.candidates if item.exact_title_match)

    def as_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "query": self.query,
            "retrieval_method": self.retrieval_method,
            "used_flexisearch": self.used_flexisearch,
            "attempted_endpoints": list(self.attempted_endpoints),
            "candidates": [
                item.as_dict(include_payload=include_payloads)
                for item in self.candidates
            ],
        }
        if include_payloads:
            result["payload"] = self.payload
        return result


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
            should_redact = (
                SENSITIVE_KEY_RE.search(key) is not None
                or key.casefold() in PRIVATE_QUERY_KEYS
            )
            safe_query.append((key, "[REDACTED]" if should_redact else value))
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


def _plain_text(value: Any, language: str) -> str:
    text = _localized_text(value, language)
    text = HTML_TAG_RE.sub(" ", html.unescape(text))
    return " ".join(text.split())


def _normalized_exact_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def _redact_query_echoes(value: Any) -> Any:
    """Remove query echoes before persisting a search/autocode response."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z_]", "", str(key).casefold())
            if normalized_key in PRIVATE_RESPONSE_KEYS:
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _redact_query_echoes(item)
        return sanitized
    if isinstance(value, list):
        return [_redact_query_echoes(item) for item in value]
    return value


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
        sensitive_values: tuple[str, ...] = (),
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
                    sensitive_values=self._sensitive_values(
                        access_token, *sensitive_values
                    ),
                ) from exc
            except OSError as exc:
                raise ICDAPIError(
                    "request setup failed",
                    endpoint=endpoint,
                    response_summary=f"{type(exc).__name__}: {exc}",
                    sensitive_values=self._sensitive_values(
                        access_token, *sensitive_values
                    ),
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
                    sensitive_values=self._sensitive_values(
                        access_token, *sensitive_values
                    ),
                )
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise ICDAPIError(
                    "response was not valid JSON",
                    endpoint=endpoint,
                    status=status,
                    response_summary=response_text,
                    sensitive_values=self._sensitive_values(
                        access_token, *sensitive_values
                    ),
                ) from exc
            if not isinstance(payload, dict):
                raise ICDAPIError(
                    "response JSON was not an object",
                    endpoint=endpoint,
                    status=status,
                    response_summary=response_text,
                    sensitive_values=self._sensitive_values(
                        access_token, *sensitive_values
                    ),
                )
            return payload
        raise AssertionError("request retry loop exhausted unexpectedly")

    def _authenticated_get_json(
        self,
        endpoint: str,
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        try:
            return self._request_json(
                "GET",
                endpoint,
                authenticated=True,
                sensitive_values=sensitive_values,
            )
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
            return self._request_json(
                "GET",
                endpoint,
                authenticated=True,
                sensitive_values=sensitive_values,
            )

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
                os.chmod(temporary, 0o600)
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

    def _query_cache_identity(self, endpoint: str) -> str:
        """Return a stable cache identity that contains no raw query text."""

        parsed = urlparse(endpoint)
        safe_query: list[tuple[str, str]] = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key.casefold() in PRIVATE_QUERY_KEYS:
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                safe_query.append((f"{key}_sha256", digest))
            else:
                safe_query.append((key, value))
        return urlunparse(parsed._replace(query=urlencode(safe_query)))

    def _cached_query_get_json(
        self,
        endpoint: str,
        *,
        query_text: str,
        kind: str,
        force_refresh: bool,
    ) -> dict[str, Any]:
        cache_identity = self._query_cache_identity(endpoint)
        path = self._cache_path(kind, cache_identity)
        if not force_refresh:
            cached = self._read_cache(path, cache_identity)
            if cached is not None:
                self._validate_discovery_payload(
                    cached,
                    endpoint=endpoint,
                    query_text=query_text,
                )
                return cached
        payload = self._authenticated_get_json(
            endpoint, sensitive_values=(query_text,)
        )
        self._validate_discovery_payload(
            payload,
            endpoint=endpoint,
            query_text=query_text,
        )
        sanitized = _redact_query_echoes(payload)
        if not isinstance(sanitized, dict):
            raise AssertionError("query response sanitizer must return an object")
        self._write_cache(path, cache_identity, sanitized)
        return sanitized

    def _discovery_text(
        self, value: str, *, name: str, max_length: int
    ) -> str:
        if not isinstance(value, str):
            raise ICDAPIError(
                f"invalid {name}",
                endpoint=self.config.base_url,
                response_summary="must be a string",
            )
        raw = value
        if any(
            unicodedata.category(character) == "Cc"
            and character not in {"\t", "\n", "\r"}
            for character in raw
        ):
            raise ICDAPIError(
                f"invalid {name}",
                endpoint=self.config.base_url,
                response_summary="control characters are not allowed",
            )
        normalized = " ".join(raw.split())
        if not normalized or len(normalized) > max_length:
            raise ICDAPIError(
                f"invalid {name}",
                endpoint=self.config.base_url,
                response_summary=f"must contain 1-{max_length} characters",
            )
        return normalized

    def _validate_discovery_get_endpoint(self, endpoint: str) -> str:
        encoded_length = len(endpoint.encode("utf-8"))
        if encoded_length > MAX_DISCOVERY_GET_URL_BYTES:
            raise ICDAPIError(
                "discovery input is too long for a GET request",
                endpoint=endpoint,
                response_summary=(
                    f"encoded URL is {encoded_length} bytes; maximum is "
                    f"{MAX_DISCOVERY_GET_URL_BYTES}"
                ),
            )
        return endpoint

    def _search_endpoint(self, query: str, *, use_flexisearch: bool) -> str:
        parameters = urlencode(
            {
                "q": query,
                "useFlexisearch": str(use_flexisearch).lower(),
                "flatResults": "true",
                "highlightingEnabled": "false",
                "medicalCodingMode": "true",
            }
        )
        endpoint = (
            f"{self.config.base_url}/icd/release/11/{self.config.release}"
            f"/mms/search?{parameters}"
        )
        return self._validate_discovery_get_endpoint(endpoint)

    def _autocode_endpoint(self, text: str) -> str:
        parameters = urlencode({"searchText": text})
        endpoint = (
            f"{self.config.base_url}/icd/release/11/{self.config.release}"
            f"/mms/autocode?{parameters}"
        )
        return self._validate_discovery_get_endpoint(endpoint)

    def _validate_discovery_payload(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str,
        query_text: str,
    ) -> None:
        error = payload.get("error")
        if error not in (None, False, "", 0):
            raise ICDAPIError(
                "discovery response reported an error",
                endpoint=endpoint,
                status=200,
                response_summary=json.dumps(payload, ensure_ascii=False),
                sensitive_values=self._sensitive_values(query_text),
            )

    def _safe_candidate_uri(self, value: Any) -> str:
        if not isinstance(value, str) or not value:
            return ""
        parsed = urlparse(value)
        expected = urlparse(self.config.base_url)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.hostname.lower() != expected.hostname.lower()
        ):
            return ""
        return value

    @staticmethod
    def _candidate_items(
        payload: dict[str, Any], retrieval_method: str
    ) -> list[dict[str, Any]]:
        if payload.get("found") is False:
            return []
        for key in ("destinationEntities", "results", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for key in ("destinationEntity", "bestMatch", "result"):
            value = payload.get(key)
            if isinstance(value, dict):
                return [value]
        if retrieval_method == "autocode" and any(
            key in payload
            for key in (
                "theCode",
                "code",
                "matchingText",
                "bestMatchingText",
                "linearizationURI",
                "linearizationUri",
            )
        ):
            return [payload]
        return []

    def _parse_candidates(
        self,
        payload: dict[str, Any],
        *,
        query: str,
        retrieval_method: str,
        used_flexisearch: bool,
        endpoint: str,
    ) -> tuple[ICDSearchCandidate, ...]:
        candidates: list[ICDSearchCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        safe_endpoint = _safe_endpoint(endpoint)
        normalized_query = _normalized_exact_text(query)
        for item in self._candidate_items(payload, retrieval_method):
            matching_text = _plain_text(
                item.get("matchingText") or item.get("bestMatchingText"),
                self.config.language,
            )
            authoritative_title = _plain_text(
                item.get("title"), self.config.language
            )
            title = authoritative_title
            if not title:
                title = matching_text
            code_value = str(item.get("theCode") or item.get("code") or "").strip()
            code = code_value.upper() if CODE_RE.fullmatch(code_value) else ""
            entity_uri = self._safe_candidate_uri(
                item.get("linearizationURI")
                or item.get("linearizationUri")
                or item.get("id")
                or item.get("@id")
            )
            foundation_uri = self._safe_candidate_uri(
                item.get("foundationURI")
                or item.get("foundationUri")
                or item.get("foundationId")
            )
            if not title or (not code and not entity_uri):
                continue
            score: float | None = None
            score_value = item.get("matchScore", item.get("score"))
            if not isinstance(score_value, bool):
                try:
                    parsed_score = float(score_value)
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(parsed_score):
                        score = parsed_score
            identity = (entity_uri, code, _normalized_exact_text(title))
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                ICDSearchCandidate(
                    rank=len(candidates) + 1,
                    title=title,
                    code=code,
                    entity_uri=entity_uri,
                    foundation_uri=foundation_uri,
                    matching_text=matching_text,
                    match_score=score,
                    exact_title_match=(
                        bool(normalized_query)
                        and not used_flexisearch
                        and (
                            bool(authoritative_title)
                            or (
                                retrieval_method == "autocode"
                                and item.get("isTitle") is True
                            )
                        )
                        and _normalized_exact_text(title) == normalized_query
                    ),
                    retrieval_method=retrieval_method,
                    used_flexisearch=used_flexisearch,
                    endpoint=safe_endpoint,
                    payload=dict(item),
                )
            )
        return tuple(candidates)

    def search_mms(
        self,
        query: str,
        *,
        flex_fallback: bool = True,
        force_refresh: bool = False,
    ) -> ICDSearchResult:
        """Return MMS search candidates, using flexisearch only when empty.

        No returned candidate is accepted as an ICD link.  In particular,
        flexisearch results remain fuzzy candidates even if ranked first.
        """

        normalized_query = self._discovery_text(
            query,
            name="search query",
            max_length=MAX_SEARCH_QUERY_CHARACTERS,
        )
        attempted: list[str] = []
        endpoint = self._search_endpoint(
            normalized_query, use_flexisearch=False
        )
        attempted.append(endpoint)
        payload = self._cached_query_get_json(
            endpoint,
            query_text=normalized_query,
            kind="search",
            force_refresh=force_refresh,
        )
        candidates = self._parse_candidates(
            payload,
            query=normalized_query,
            retrieval_method="search",
            used_flexisearch=False,
            endpoint=endpoint,
        )
        used_flexisearch = False
        if not candidates and flex_fallback:
            endpoint = self._search_endpoint(
                normalized_query, use_flexisearch=True
            )
            attempted.append(endpoint)
            payload = self._cached_query_get_json(
                endpoint,
                query_text=normalized_query,
                kind="search-flex",
                force_refresh=force_refresh,
            )
            candidates = self._parse_candidates(
                payload,
                query=normalized_query,
                retrieval_method="search",
                used_flexisearch=True,
                endpoint=endpoint,
            )
            used_flexisearch = True
        return ICDSearchResult(
            query=normalized_query,
            retrieval_method="search",
            used_flexisearch=used_flexisearch,
            attempted_endpoints=tuple(_safe_endpoint(item) for item in attempted),
            candidates=candidates,
            payload=payload,
        )

    def autocode_mms(
        self,
        text: str,
        *,
        force_refresh: bool = False,
    ) -> ICDSearchResult:
        """Return the MMS autocode response strictly as an unverified candidate."""

        normalized_text = self._discovery_text(
            text,
            name="autocode text",
            max_length=MAX_AUTOCODE_TEXT_CHARACTERS,
        )
        endpoint = self._autocode_endpoint(normalized_text)
        payload = self._cached_query_get_json(
            endpoint,
            query_text=normalized_text,
            kind="autocode",
            force_refresh=force_refresh,
        )
        candidates = self._parse_candidates(
            payload,
            query=normalized_text,
            retrieval_method="autocode",
            used_flexisearch=False,
            endpoint=endpoint,
        )
        return ICDSearchResult(
            query=normalized_text,
            retrieval_method="autocode",
            used_flexisearch=False,
            attempted_endpoints=(_safe_endpoint(endpoint),),
            candidates=candidates,
            payload=payload,
        )

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
