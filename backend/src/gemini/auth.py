"""Gemini HMAC-SHA384 authentication header generation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def make_auth_headers(
    api_key: str,
    api_secret: str,
    request_path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build the three authentication headers required by Gemini's private API.

    The protocol:
    1. Build a JSON payload containing at minimum ``request`` (the endpoint path)
       and ``nonce`` (Unix timestamp in milliseconds).
    2. Base64-encode that JSON string.
    3. HMAC-SHA384 sign the base64 bytes using the API secret.
    4. Return the headers ``X-GEMINI-APIKEY``, ``X-GEMINI-PAYLOAD``, and
       ``X-GEMINI-SIGNATURE``.

    Parameters
    ----------
    api_key:
        The Gemini API key.
    api_secret:
        The Gemini API secret (used as HMAC key).
    request_path:
        The request path, e.g. ``/v1/prediction-markets/order``.
    payload:
        Optional additional payload fields merged into the base payload.

    Returns
    -------
    dict[str, str]
        Headers ready to be passed to an HTTP request.
    """
    nonce = str(int(time.time() * 1000))

    body: dict[str, Any] = {
        "request": request_path,
        "nonce": nonce,
    }
    if payload:
        body.update(payload)

    encoded_body = json.dumps(body).encode("utf-8")
    b64_payload = base64.b64encode(encoded_body)

    signature = hmac.new(
        api_secret.encode("utf-8"),
        b64_payload,
        hashlib.sha384,
    ).hexdigest()

    return {
        "X-GEMINI-APIKEY": api_key,
        "X-GEMINI-PAYLOAD": b64_payload.decode("utf-8"),
        "X-GEMINI-SIGNATURE": signature,
        "Content-Type": "text/plain",
    }
