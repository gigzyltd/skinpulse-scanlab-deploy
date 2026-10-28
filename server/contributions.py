"""Opt-in photo contributions (closed testing).

A photo is stored ONLY when the client sends explicit consent. Storage is a private
S3-compatible bucket (Cloudflare R2 free tier intended). Nothing is ever public: no
public URLs are generated and the bucket must not have public access. When storage
is not configured, contributions are refused (503) and nothing is kept.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol
from xml.etree import ElementTree

TESTER_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
# Requested view. left/right are accepted from older app builds.
VIEWS = {"front", "left_45", "right_45", "left", "right", "unknown"}
_EXT = {"image/png": "png", "image/webp": "webp", "image/jpeg": "jpg"}


class ContributionStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...


@dataclass(frozen=True)
class S3Config:
    endpoint: str  # e.g. https://<account>.r2.cloudflarestorage.com
    bucket: str
    access_key: str
    secret_key: str
    region: str = "auto"


Transport = Callable[[urllib.request.Request], tuple[int, bytes]]


def _urlopen(req: urllib.request.Request) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https endpoint
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class S3Store:
    """Minimal SigV4 client: PUT object, list by prefix, delete object."""

    def __init__(self, config: S3Config, transport: Transport = _urlopen, now: Callable[[], dt.datetime] | None = None):
        self.c = config
        self.transport = transport
        self.now = now or (lambda: dt.datetime.now(dt.timezone.utc))

    def _request(self, method: str, key: str, query: dict[str, str] | None = None, body: bytes = b"", content_type: str | None = None):
        host = urllib.parse.urlsplit(self.c.endpoint).netloc
        path = "/" + self.c.bucket + ("/" + urllib.parse.quote(key, safe="/-_.~") if key else "")
        q = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in sorted((query or {}).items()))
        t = self.now()
        amz = t.strftime("%Y%m%dT%H%M%SZ")
        day = t.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz}
        if content_type:
            headers["content-type"] = content_type
        signed = ";".join(sorted(headers))
        canonical = "\n".join([method, path, q, "".join(f"{k}:{headers[k]}\n" for k in sorted(headers)), signed, payload_hash])
        scope = f"{day}/{self.c.region}/s3/aws4_request"
        to_sign = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(canonical.encode()).hexdigest()])
        k = _sign(_sign(_sign(_sign(("AWS4" + self.c.secret_key).encode(), day), self.c.region), "s3"), "aws4_request")
        sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
        headers["authorization"] = f"AWS4-HMAC-SHA256 Credential={self.c.access_key}/{scope}, SignedHeaders={signed}, Signature={sig}"
        url = self.c.endpoint.rstrip("/") + path + (f"?{q}" if q else "")
        req = urllib.request.Request(url, data=body if method == "PUT" else None, method=method,
                                     headers={k2: v for k2, v in headers.items() if k2 != "host"})
        return self.transport(req)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        status, _ = self._request("PUT", key, body=data, content_type=content_type)
        if status not in (200, 201):
            raise RuntimeError(f"storage put failed: {status}")

    def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        token: str | None = None
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            status, body = self._request("GET", "", query=query)
            if status != 200:
                raise RuntimeError(f"storage list failed: {status}")
            root = ElementTree.fromstring(body)
            ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
            find = (lambda e, p: e.findall(f"s:{p}", ns)) if ns else (lambda e, p: e.findall(p))
            for item in find(root, "Contents"):
                key = (item.find("s:Key", ns) if ns else item.find("Key")).text or ""
                if not key.startswith(prefix):
                    continue
                st, _ = self._request("DELETE", key)
                if st not in (200, 204):
                    raise RuntimeError(f"storage delete failed: {st}")
                deleted += 1
            trunc = (root.find("s:IsTruncated", ns) if ns else root.find("IsTruncated"))
            nxt = (root.find("s:NextContinuationToken", ns) if ns else root.find("NextContinuationToken"))
            if trunc is not None and trunc.text == "true" and nxt is not None and nxt.text:
                token = nxt.text
                continue
            return deleted


def store_from_env(env) -> ContributionStore | None:
    names = ("SCANLAB_CONTRIB_S3_ENDPOINT", "SCANLAB_CONTRIB_S3_BUCKET", "SCANLAB_CONTRIB_S3_ACCESS_KEY", "SCANLAB_CONTRIB_S3_SECRET_KEY")
    values = [str(env.get(n, "")).strip() for n in names]
    if not all(values) or not values[0].startswith("https://"):
        return None
    return S3Store(S3Config(*values, region=str(env.get("SCANLAB_CONTRIB_S3_REGION", "auto")).strip() or "auto"))


def parse_meta(header: str | None) -> dict:
    """X-Contribution-Meta: base64url(JSON). Consent must be literally true."""
    from scanlab.server.errors import ApiError

    if not header:
        raise ApiError(400, "consent_required", "Explicit consent is required to store a photo.")
    try:
        pad = "=" * (-len(header) % 4)
        meta = json.loads(base64.urlsafe_b64decode(header + pad))
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "invalid_request", "Contribution metadata is not valid.") from exc
    if not isinstance(meta, dict) or meta.get("consent") is not True:
        raise ApiError(400, "consent_required", "Explicit consent is required to store a photo.")
    tester = meta.get("tester_id")
    if not isinstance(tester, str) or not TESTER_ID.match(tester):
        raise ApiError(400, "invalid_request", "tester_id is invalid.")
    for name in ("consent_version", "consented_at"):
        if not isinstance(meta.get(name), str) or not meta[name] or len(meta[name]) > 64:
            raise ApiError(400, "invalid_request", f"{name} is required.")
    view = meta.get("view") if meta.get("view") in VIEWS else "unknown"
    pose = _pose(meta.get("estimated_pose"))
    tz = meta.get("tz_offset_minutes")
    tz = tz if isinstance(tz, int) and not isinstance(tz, bool) and -840 <= tz <= 840 else None
    result = meta.get("result")
    if result is not None and (not isinstance(result, dict) or len(json.dumps(result)) > 20_000):
        raise ApiError(400, "invalid_request", "result is invalid.")
    return {
        "tester_id": tester,
        "view": view,
        "consent": True,
        "consent_version": meta["consent_version"],
        "consented_at": meta["consented_at"],
        "estimated_pose": pose,
        "client_captured_at": str(meta.get("captured_at", ""))[:64],
        "client_tz_offset_minutes": tz,
        "result": result,
    }


def _pose(value) -> dict | None:
    """Server-estimated pose proxy echoed by the app: three finite degrees, else None."""
    if not isinstance(value, dict):
        return None
    out = {}
    for name in ("yaw_deg", "pitch_deg", "roll_deg"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or abs(item) > 180:
            return None
        out[name] = round(float(item), 1)
    return out


def object_keys(tester_id: str, media_type: str, when: dt.datetime) -> tuple[str, str]:
    base = f"contributions/{tester_id}/{when.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    return f"{base}.{_EXT.get(media_type, 'jpg')}", f"{base}.json"
