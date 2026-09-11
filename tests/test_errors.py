from fastapi import HTTPException

from dsg_compliance.api.errors import friendly_llm_error


def test_rate_limit_by_exception_name_maps_to_503():
    class GLMRateLimitError(Exception):
        pass

    result = friendly_llm_error(GLMRateLimitError("slow down"))
    assert isinstance(result, HTTPException)
    assert result.status_code == 503
    assert "Ratenlimit" in result.detail


def test_429_in_message_maps_to_503():
    result = friendly_llm_error(Exception("Error code: 429 Too Many Requests"))
    assert result.status_code == 503


def test_quota_keyword_case_insensitive_maps_to_503():
    result = friendly_llm_error(Exception("Daily QUOTA exceeded"))
    assert result.status_code == 503


def test_missing_api_key_runtime_error_maps_to_500_with_config_hint():
    result = friendly_llm_error(RuntimeError("GEMINI_API_KEY not set in .env"))
    assert result.status_code == 500
    assert "Konfigurationsfehler" in result.detail
    assert "API_KEY" in result.detail


def test_api_key_message_without_runtime_error_type_is_not_treated_as_config_error():
    # only RuntimeError is special-cased for API_KEY - a different exception
    # type with "API_KEY" in the message should fall through to the generic branch
    result = friendly_llm_error(ValueError("bad API_KEY format"))
    assert result.status_code == 500
    assert "Unerwarteter Fehler" in result.detail


def test_unrecognized_exception_maps_to_generic_500():
    result = friendly_llm_error(ConnectionError("boom"))
    assert result.status_code == 500
    assert "ConnectionError" in result.detail
    assert "boom" in result.detail


def test_long_message_is_truncated_to_200_chars():
    result = friendly_llm_error(Exception("x" * 500))
    assert len(result.detail) <= len("Unerwarteter Fehler (Exception): ") + 200
