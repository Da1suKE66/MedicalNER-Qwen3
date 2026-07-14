from __future__ import annotations

import base64
import json
import ssl
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.icd_api import (  # noqa: E402
    ICDAPIConfig,
    ICDAPIConfigurationError,
    ICDCodeResult,
    ICDAPIError,
    ICDSearchCandidate,
    ICDSearchResult,
    WHOICDClient,
    _UrllibSession,
    _clear_process_token_cache_for_tests,
)
import scripts.validate_icd_codes as validate_cli  # noqa: E402
from scripts.validate_icd_codes import extract_code_references  # noqa: E402


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if text is None else text
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class FakeUrlopenResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, url: str, body: bytes) -> None:
        self._url = url
        self._body = body

    def geturl(self) -> str:
        return self._url

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeUrlopenResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class FakeOpener:
    def __init__(self) -> None:
        self.request: Any = None
        self.timeout: float | None = None

    def open(self, request: Any, timeout: float) -> FakeUrlopenResponse:
        self.request = request
        self.timeout = timeout
        return FakeUrlopenResponse(
            request.full_url,
            b'{"access_token":"transport-token","expires_in":3600}',
        )


def token_response(token: str = "test-access-token") -> FakeResponse:
    return FakeResponse(
        200,
        {"access_token": token, "expires_in": 3600, "token_type": "Bearer"},
    )


def codeinfo_response(code: str = "6A05.0") -> FakeResponse:
    return FakeResponse(
        200,
        {
            "@id": (
                "http://id.who.int/icd/release/11/2025-01/mms/"
                f"codeinfo/{code}"
            ),
            "code": code,
            "stemCode": code,
            "stemId": "http://id.who.int/icd/release/11/2025-01/mms/821852937",
        },
    )


def entity_response() -> FakeResponse:
    return FakeResponse(
        200,
        {
            "@id": "http://id.who.int/icd/release/11/2025-01/mms/821852937",
            "title": {"@language": "en", "@value": "Attention deficit hyperactivity disorder"},
            "code": "6A05.0",
        },
    )


class WHOICDClientTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_process_token_cache_for_tests()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name) / "icd-cache"
        self.config = ICDAPIConfig(
            client_id="unit-client-id",
            client_secret="unit-client-secret",
            release="2025-01",
            language="en",
            cache_dir=self.cache_dir,
            timeout_seconds=7.5,
            max_retries=2,
            retry_backoff_seconds=0.01,
        )

    def test_lookup_uses_oauth_v2_tls_and_persistent_response_cache(self) -> None:
        session = FakeSession([token_response(), codeinfo_response(), entity_response()])
        client = WHOICDClient(self.config, session=session, sleep=lambda _delay: None)

        first = client.lookup_code("6a05.0")
        second = client.lookup_code("6A05.0")

        self.assertEqual(first.title, "Attention deficit hyperactivity disorder")
        self.assertEqual(second.entity_uri, first.entity_uri)
        self.assertEqual(len(session.calls), 3, "second lookup should use disk cache")

        token_call, codeinfo_call, entity_call = session.calls
        self.assertEqual(token_call["method"], "POST")
        self.assertEqual(token_call["auth"], ("unit-client-id", "unit-client-secret"))
        self.assertEqual(
            token_call["data"],
            {"grant_type": "client_credentials", "scope": "icdapi_access"},
        )
        self.assertTrue(token_call["verify"])
        self.assertEqual(token_call["timeout"], 7.5)

        for call in (codeinfo_call, entity_call):
            self.assertEqual(call["headers"]["API-Version"], "v2")
            self.assertEqual(call["headers"]["Accept-Language"], "en")
            self.assertEqual(call["headers"]["Accept"], "application/json")
            self.assertEqual(
                call["headers"]["Authorization"], "Bearer test-access-token"
            )
            self.assertTrue(call["verify"])
            self.assertTrue(call["url"].startswith("https://"))
        self.assertIn("/mms/codeinfo/6A05.0", codeinfo_call["url"])
        self.assertEqual(
            entity_call["url"],
            "https://id.who.int/icd/release/11/2025-01/mms/821852937",
        )

        cache_files = sorted(self.cache_dir.glob("*.json"))
        self.assertEqual(len(cache_files), 2)
        self.assertTrue(
            all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in cache_files)
        )
        cache_text = "\n".join(path.read_text(encoding="utf-8") for path in cache_files)
        self.assertNotIn("unit-client-secret", cache_text)
        self.assertNotIn("test-access-token", cache_text)
        self.assertNotIn("Authorization", cache_text)

    def test_stdlib_transport_encodes_basic_auth_and_client_credentials_form(self) -> None:
        transport = _UrllibSession(ssl.create_default_context())
        opener = FakeOpener()
        transport._opener = opener
        response = transport.request(
            "POST",
            "https://icdaccessmanagement.who.int/connect/token",
            headers={"Accept": "application/json"},
            data={"grant_type": "client_credentials", "scope": "icdapi_access"},
            auth=("client-id", "client-secret"),
            timeout=4.0,
            verify=True,
            allow_redirects=True,
        )

        expected_basic = base64.b64encode(b"client-id:client-secret").decode("ascii")
        self.assertEqual(
            opener.request.get_header("Authorization"), f"Basic {expected_basic}"
        )
        self.assertEqual(
            opener.request.get_header("Content-type"),
            "application/x-www-form-urlencoded",
        )
        self.assertEqual(
            opener.request.data,
            b"grant_type=client_credentials&scope=icdapi_access",
        )
        self.assertEqual(opener.timeout, 4.0)
        self.assertEqual(response.json()["access_token"], "transport-token")

    def test_process_token_is_reused_across_clients(self) -> None:
        first_session = FakeSession(
            [token_response("shared-token"), codeinfo_response(), entity_response()]
        )
        WHOICDClient(
            self.config, session=first_session, sleep=lambda _delay: None
        ).lookup_code("6A05.0", force_refresh=True)

        second_session = FakeSession([codeinfo_response(), entity_response()])
        WHOICDClient(
            self.config, session=second_session, sleep=lambda _delay: None
        ).lookup_code("6A05.0", force_refresh=True)

        self.assertEqual(len(second_session.calls), 2)
        self.assertTrue(all(call["method"] == "GET" for call in second_session.calls))
        self.assertTrue(
            all(
                call["headers"]["Authorization"] == "Bearer shared-token"
                for call in second_session.calls
            )
        )

    def test_retryable_http_error_is_retried(self) -> None:
        sleeps: list[float] = []
        session = FakeSession(
            [
                token_response(),
                FakeResponse(503, {"error": "temporarily unavailable"}),
                codeinfo_response(),
                entity_response(),
            ]
        )
        result = WHOICDClient(
            self.config, session=session, sleep=sleeps.append
        ).lookup_code("6A05.0", force_refresh=True)

        self.assertEqual(result.canonical_code, "6A05.0")
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0)
        self.assertEqual(len(session.calls), 4)

    def test_timeout_is_retried_and_final_error_has_safe_endpoint(self) -> None:
        config = ICDAPIConfig(
            client_id="unit-client-id",
            client_secret="unit-client-secret",
            cache_dir=self.cache_dir,
            max_retries=1,
            retry_backoff_seconds=0,
        )
        session = FakeSession(
            [TimeoutError("timeout one"), TimeoutError("timeout two")]
        )
        with self.assertRaises(ICDAPIError) as raised:
            WHOICDClient(config, session=session, sleep=lambda _delay: None).lookup_code(
                "6A05.0"
            )
        message = str(raised.exception)
        self.assertIn("endpoint=https://icdaccessmanagement.who.int/connect/token", message)
        self.assertIn("HTTP status=unavailable", message)
        self.assertNotIn("unit-client-secret", message)

    def test_401_refreshes_process_cached_token_once(self) -> None:
        session = FakeSession(
            [
                token_response("revoked-token"),
                FakeResponse(401, {"error": "invalid_token"}),
                token_response("fresh-token"),
                codeinfo_response(),
                entity_response(),
            ]
        )
        result = WHOICDClient(
            self.config, session=session, sleep=lambda _delay: None
        ).lookup_code("6A05.0", force_refresh=True)

        self.assertEqual(result.title, "Attention deficit hyperactivity disorder")
        token_calls = [call for call in session.calls if call["method"] == "POST"]
        self.assertEqual(len(token_calls), 2)
        final_gets = [call for call in session.calls if call["method"] == "GET"][-2:]
        self.assertTrue(
            all(
                call["headers"]["Authorization"] == "Bearer fresh-token"
                for call in final_gets
            )
        )

    def test_error_redacts_secret_and_token_but_keeps_status_and_endpoint(self) -> None:
        response_text = json.dumps(
            {
                "error": "invalid_client",
                "client_secret": "unit-client-secret",
                "access_token": "server-echoed-token",
            }
        )
        session = FakeSession(
            [FakeResponse(401, {}, text=response_text)]
        )
        with self.assertRaises(ICDAPIError) as raised:
            WHOICDClient(
                self.config, session=session, sleep=lambda _delay: None
            ).lookup_code("6A05.0")

        error = raised.exception
        rendered = str(error)
        self.assertIn("endpoint=https://icdaccessmanagement.who.int/connect/token", rendered)
        self.assertIn("HTTP status=401", rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertNotIn("unit-client-secret", rendered)
        self.assertNotIn("server-echoed-token", rendered)
        self.assertEqual(error.as_dict()["http_status"], 401)

    def test_non_tls_configuration_is_rejected(self) -> None:
        with self.assertRaises(ICDAPIConfigurationError):
            ICDAPIConfig(
                client_id="id",
                client_secret="secret",
                base_url="http://id.who.int",
            )
        with self.assertRaises(ICDAPIConfigurationError):
            ICDAPIConfig(
                client_id="id",
                client_secret="secret",
                token_url="http://auth.example/token",
            )

    def test_from_env_expands_nested_cache_path_references(self) -> None:
        config = ICDAPIConfig.from_env(
            {
                "WHO_ICD_CLIENT_ID": "id",
                "WHO_ICD_CLIENT_SECRET": "secret",
                "WHO_ICD_RELEASE": "2025-01",
                "WHO_ICD_LANGUAGE": "en",
                "KG_CACHE_ROOT": "/cache/unit-test",
                "ICD_API_CACHE_DIR": (
                    "${KG_CACHE_ROOT}/icd-api/${WHO_ICD_RELEASE}/"
                    "${WHO_ICD_LANGUAGE}"
                ),
            }
        )

        self.assertEqual(
            config.cache_dir,
            Path("/cache/unit-test/icd-api/2025-01/en"),
        )

    def test_from_env_rejects_unresolved_cache_path_reference(self) -> None:
        with self.assertRaisesRegex(
            ICDAPIConfigurationError,
            "undefined environment variable MISSING_CACHE_ROOT",
        ):
            ICDAPIConfig.from_env(
                {
                    "WHO_ICD_CLIENT_ID": "id",
                    "WHO_ICD_CLIENT_SECRET": "secret",
                    "ICD_API_CACHE_DIR": "${MISSING_CACHE_ROOT}/icd-api",
                }
            )

    def test_codeinfo_without_stem_id_is_reported_with_http_status(self) -> None:
        session = FakeSession(
            [token_response(), FakeResponse(200, {"code": "6A05.0"})]
        )
        with self.assertRaises(ICDAPIError) as raised:
            WHOICDClient(
                self.config, session=session, sleep=lambda _delay: None
            ).lookup_code("6A05.0", force_refresh=True)
        self.assertEqual(raised.exception.status, 200)
        self.assertIn("codeinfo", raised.exception.endpoint)
        self.assertIn("stemId", str(raised.exception))

    def test_search_uses_regular_then_flex_and_caches_without_raw_query(self) -> None:
        query = "rare diagnostic phrase"
        regular = FakeResponse(
            200,
            {
                "error": False,
                "words": ["rare", "diagnostic", "phrase"],
                "destinationEntities": [],
            },
        )
        flex = FakeResponse(
            200,
            {
                "error": False,
                "destinationEntities": [
                    {
                        "id": (
                            "http://id.who.int/icd/release/11/2025-01/"
                            "mms/123456789"
                        ),
                        "title": "Rare diagnostic disorder",
                        "theCode": "6E8Y",
                        "score": 0.72,
                    }
                ],
            },
        )
        session = FakeSession([token_response(), regular, flex])
        client = WHOICDClient(
            self.config, session=session, sleep=lambda _delay: None
        )

        first = client.search_mms(query)
        second = client.search_mms(query)

        self.assertIsInstance(first, ICDSearchResult)
        self.assertTrue(first.used_flexisearch)
        self.assertEqual(second.candidates, first.candidates)
        self.assertEqual(len(first.attempted_endpoints), 2)
        self.assertEqual(len(first.candidates), 1)
        candidate = first.candidates[0]
        self.assertIsInstance(candidate, ICDSearchCandidate)
        self.assertEqual(candidate.code, "6E8Y")
        self.assertFalse(candidate.exact_title_match)
        self.assertTrue(candidate.requires_code_validation)

        self.assertEqual(len(session.calls), 3, "second search should use disk cache")
        regular_call, flex_call = session.calls[1:]
        regular_query = parse_qs(urlparse(regular_call["url"]).query)
        flex_query = parse_qs(urlparse(flex_call["url"]).query)
        self.assertEqual(regular_query["q"], [query])
        self.assertEqual(regular_query["useFlexisearch"], ["false"])
        self.assertEqual(flex_query["useFlexisearch"], ["true"])
        self.assertEqual(regular_query["flatResults"], ["true"])
        self.assertEqual(regular_query["highlightingEnabled"], ["false"])
        self.assertEqual(regular_query["medicalCodingMode"], ["true"])

        cache_files = sorted(self.cache_dir.glob("*.json"))
        self.assertEqual(len(cache_files), 2)
        cache_text = "\n".join(
            path.read_text(encoding="utf-8") for path in cache_files
        )
        self.assertNotIn(query, cache_text)
        self.assertNotIn("unit-client-secret", cache_text)
        self.assertNotIn("test-access-token", cache_text)
        self.assertIn("q_sha256", cache_text)
        self.assertIn('"words": "[REDACTED]"', cache_text)

    def test_flex_result_is_never_labeled_as_an_exact_title_candidate(self) -> None:
        query = "Exact example disorder"
        regular = FakeResponse(
            200,
            {"error": False, "destinationEntities": []},
        )
        flex = FakeResponse(
            200,
            {
                "error": False,
                "destinationEntities": [
                    {
                        "id": (
                            "http://id.who.int/icd/release/11/2025-01/"
                            "mms/123456789"
                        ),
                        "title": query,
                        "theCode": "6E8Y",
                    }
                ],
            },
        )
        session = FakeSession([token_response(), regular, flex])

        result = WHOICDClient(
            self.config,
            session=session,
            sleep=lambda _delay: None,
        ).search_mms(query)

        self.assertTrue(result.used_flexisearch)
        self.assertTrue(result.candidates[0].used_flexisearch)
        self.assertFalse(result.candidates[0].exact_title_match)
        self.assertEqual(result.exact_title_candidates, ())

    def test_success_status_error_payload_is_rejected_before_cache_write(self) -> None:
        query = "private diagnostic phrase"
        session = FakeSession(
            [
                token_response(),
                FakeResponse(
                    200,
                    {
                        "error": True,
                        "errorMessage": f"invalid query: {query}",
                    },
                ),
            ]
        )

        with self.assertRaises(ICDAPIError) as raised:
            WHOICDClient(
                self.config,
                session=session,
                sleep=lambda _delay: None,
            ).search_mms(query)

        self.assertNotIn(query, str(raised.exception))
        self.assertEqual(list(self.cache_dir.glob("*.json")), [])

    def test_discovery_inputs_and_encoded_get_urls_are_bounded(self) -> None:
        session = FakeSession([])
        client = WHOICDClient(self.config, session=session)

        with self.assertRaisesRegex(ICDAPIError, "invalid search query"):
            client.search_mms("x" * 513)
        with self.assertRaisesRegex(ICDAPIError, "invalid autocode text"):
            client.autocode_mms("x" * 1_001)
        with self.assertRaisesRegex(ICDAPIError, "too long for a GET request"):
            client.search_mms("病" * 250)
        with self.assertRaisesRegex(ICDAPIError, "too long for a GET request"):
            client.autocode_mms("病" * 250)

        self.assertEqual(session.calls, [])

    def test_search_stops_after_regular_results_and_marks_exact_title_only(self) -> None:
        query = "  attention   DEFICIT hyperactivity disorder  "
        response = FakeResponse(
            200,
            {
                "error": False,
                "destinationEntities": [
                    {
                        "id": (
                            "http://id.who.int/icd/release/11/2025-01/"
                            "mms/821852937"
                        ),
                        "title": (
                            "Attention deficit hyperactivity disorder"
                        ),
                        "theCode": "6A05.0",
                    },
                    {
                        "id": "https://untrusted.example/entity/1",
                        "title": "Attention deficit disorder",
                        "theCode": "not a valid code!",
                    },
                ],
            },
        )
        session = FakeSession([token_response(), response])
        result = WHOICDClient(
            self.config, session=session, sleep=lambda _delay: None
        ).search_mms(query)

        self.assertFalse(result.used_flexisearch)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.exact_title_candidates), 1)
        candidate = result.exact_title_candidates[0]
        self.assertEqual(candidate.code, "6A05.0")
        self.assertTrue(candidate.exact_title_match)
        serialized = candidate.as_dict()
        self.assertTrue(serialized["requires_code_validation"])
        self.assertNotIn("accepted", serialized)
        self.assertIn("q=%5BREDACTED%5D", candidate.endpoint)

    def test_autocode_returns_unverified_candidate_and_hides_input_in_cache(self) -> None:
        text = "Patient with cerebrovascular accident"
        response = FakeResponse(
            200,
            {
                "searchText": text,
                "matchingText": "Cerebrovascular accident",
                "theCode": "8B20",
                "foundationURI": "http://id.who.int/icd/entity/123456",
                "linearizationURI": (
                    "http://id.who.int/icd/release/11/2025-01/mms/123456"
                ),
                "matchScore": 0.98,
                "isTitle": True,
            },
        )
        session = FakeSession([token_response(), response])
        result = WHOICDClient(
            self.config, session=session, sleep=lambda _delay: None
        ).autocode_mms(text)

        self.assertEqual(result.retrieval_method, "autocode")
        self.assertFalse(result.used_flexisearch)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.code, "8B20")
        self.assertEqual(candidate.match_score, 0.98)
        self.assertFalse(candidate.exact_title_match)
        self.assertTrue(candidate.requires_code_validation)
        self.assertNotIn("accepted", result.as_dict()["candidates"][0])

        autocode_call = session.calls[1]
        parsed = urlparse(autocode_call["url"])
        self.assertTrue(parsed.path.endswith("/mms/autocode"))
        self.assertEqual(parse_qs(parsed.query)["searchText"], [text])
        self.assertIn("searchText=%5BREDACTED%5D", result.attempted_endpoints[0])

        cache_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in self.cache_dir.glob("*.json")
        )
        self.assertNotIn(text, cache_text)
        self.assertIn("searchText_sha256", cache_text)
        self.assertIn('"searchText": "[REDACTED]"', cache_text)

    def test_autocode_synonym_match_is_not_labeled_exact_title(self) -> None:
        text = "heart attack"
        response = FakeResponse(
            200,
            {
                "searchText": text,
                "matchingText": text,
                "theCode": "BA41.Z",
                "linearizationURI": (
                    "http://id.who.int/icd/release/11/2025-01/mms/987654"
                ),
                "isTitle": False,
            },
        )
        session = FakeSession([token_response(), response])
        result = WHOICDClient(
            self.config, session=session, sleep=lambda _delay: None
        ).autocode_mms(text)

        self.assertEqual(result.candidates[0].title, text)
        self.assertFalse(result.candidates[0].exact_title_match)
        self.assertEqual(result.exact_title_candidates, ())

    def test_discovery_error_redacts_query_from_endpoint_and_response(self) -> None:
        query = "private diagnostic text"
        session = FakeSession(
            [
                token_response(),
                FakeResponse(
                    400,
                    {"error": "invalid query", "query": query},
                ),
            ]
        )
        with self.assertRaises(ICDAPIError) as raised:
            WHOICDClient(
                self.config, session=session, sleep=lambda _delay: None
            ).search_mms(query)
        message = str(raised.exception)
        self.assertNotIn(query, message)
        self.assertIn("q=%5BREDACTED%5D", message)

    def test_discovery_rejects_blank_and_non_whitespace_control_text(self) -> None:
        client = WHOICDClient(self.config, session=FakeSession([]))
        with self.assertRaises(ICDAPIError):
            client.search_mms("   ")
        with self.assertRaises(ICDAPIError):
            client.autocode_mms("diagnosis\x00hidden")


class CodeExtractionTests(unittest.TestCase):
    def test_extracts_raw_crawler_records(self) -> None:
        payload = {
            "metadata": {"total_count": 2},
            "entities": [
                {"id": "uri:one", "code": "6A05.0", "title": "ADHD"},
                {"id": "uri:two", "code": "6A05.0", "title": "ADHD"},
            ],
        }
        input_format, codes = extract_code_references(payload)
        self.assertEqual(input_format, "raw")
        self.assertEqual(len(codes), 1)
        self.assertEqual(codes[0]["code"], "6A05.0")
        self.assertEqual(len(codes[0]["references"]), 2)
        self.assertEqual(codes[0]["expected_titles"], ["ADHD"])

    def test_extracts_schema_v2_source_and_entity_codes(self) -> None:
        payload = [
            {
                "schema_version": "2.0.0-draft.1",
                "source_record_id": "uri:one",
                "source_code": "6A05.0",
                "source_title": "Attention deficit hyperactivity disorder",
                "output": {
                    "entities": [
                        {
                            "id": "D1",
                            "label": "Disease",
                            "name": "Attention deficit hyperactivity disorder",
                            "properties": {"icdcode": "6A05.0"},
                        },
                        {
                            "id": "D2",
                            "label": "Disease",
                            "name": "Anxiety disorder",
                            "properties": {"icdcode": "6B00"},
                        },
                    ],
                    "relations": [],
                },
            }
        ]
        input_format, codes = extract_code_references(payload)
        self.assertEqual(input_format, "schema-v2")
        self.assertEqual([item["code"] for item in codes], ["6A05.0", "6B00"])
        self.assertEqual(len(codes[0]["references"]), 2)
        self.assertIn(".properties.icdcode", codes[1]["references"][0]["path"])

    def test_extracts_codes_removed_pending_non_main_entity_linking(self) -> None:
        payload = [
            {
                "schema_version": "2.0.0-draft.2",
                "source_record_id": "uri:one",
                "source_code": "6A05.0",
                "source_title": "Canonical disorder",
                "migration": {
                    "unverified_codes": [
                        {"entity_id": "D2", "property": "DSM-5 Code", "value": "6B00"}
                    ]
                },
                "output": {
                    "entities": [
                        {
                            "id": "D1",
                            "label": "Disease",
                            "name": "Canonical disorder",
                            "properties": {"icdcode": "6A05.0"},
                        },
                        {
                            "id": "D2",
                            "label": "Disease",
                            "name": "Anxiety disorder",
                            "properties": {},
                        },
                    ],
                    "relations": [],
                },
            }
        ]
        input_format, codes = extract_code_references(payload)
        self.assertEqual(input_format, "schema-v2")
        self.assertEqual([item["code"] for item in codes], ["6A05.0", "6B00"])
        pending = codes[1]
        self.assertEqual(pending["expected_titles"], ["Anxiety disorder"])
        self.assertIn("migration.unverified_codes", pending["references"][0]["path"])


class ValidationCLITests(unittest.TestCase):
    def test_smoke_cli_writes_compact_json_report_without_loading_real_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            config = ICDAPIConfig(
                client_id="cli-id",
                client_secret="cli-secret",
                cache_dir=Path(directory) / "cache",
            )
            lookup = ICDCodeResult(
                requested_code="6A05.0",
                canonical_code="6A05.0",
                stem_code="6A05.0",
                entity_uri=(
                    "http://id.who.int/icd/release/11/2025-01/mms/821852937"
                ),
                title="Attention deficit hyperactivity disorder",
                release="2025-01",
                language="en",
                codeinfo_endpoint=(
                    "https://id.who.int/icd/release/11/2025-01/mms/"
                    "codeinfo/6A05.0"
                ),
                entity_endpoint=(
                    "https://id.who.int/icd/release/11/2025-01/mms/821852937"
                ),
                codeinfo={
                    "code": "6A05.0",
                    "stemCode": "6A05.0",
                    "stemId": (
                        "http://id.who.int/icd/release/11/2025-01/mms/821852937"
                    ),
                },
                entity={
                    "title": {
                        "@value": "Attention deficit hyperactivity disorder"
                    }
                },
            )

            class StubClient:
                def __init__(self, received_config: ICDAPIConfig) -> None:
                    self.config = received_config

                def __enter__(self) -> "StubClient":
                    return self

                def __exit__(self, *_args: Any) -> None:
                    return None

                def lookup_code(
                    self, code: str, *, force_refresh: bool = False
                ) -> ICDCodeResult:
                    self.last_call = (code, force_refresh)
                    return lookup

            with (
                patch.object(validate_cli, "_load_local_environment") as load_env,
                patch.object(validate_cli.ICDAPIConfig, "from_env", return_value=config),
                patch.object(validate_cli, "WHOICDClient", StubClient),
            ):
                exit_code = validate_cli.main(
                    ["--smoke-code", "6A05.0", "--output", str(output)]
                )

            load_env.assert_called_once_with()
            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "smoke")
            self.assertEqual(report["summary"]["valid_codes"], 1)
            self.assertEqual(report["summary"]["api_errors"], 0)
            self.assertEqual(report["results"][0]["title"], lookup.title)
            rendered = output.read_text(encoding="utf-8")
            self.assertNotIn("cli-secret", rendered)
            self.assertNotIn("access_token", rendered)


if __name__ == "__main__":
    unittest.main()
