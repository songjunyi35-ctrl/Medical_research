"""Print non-sensitive Faspex pagination fields for two UTSW directories."""
from __future__ import annotations

import base64
import html
import json
import re
from urllib.parse import parse_qs, urlparse

import requests


BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/utsw-glioma/"


def main() -> None:
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    link = html.unescape(next(value for value in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in value))
    state = parse_qs(urlparse(link).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if int(context["id"]) != 1265:
        raise RuntimeError("unexpected package")
    config = session.get(f"{BASE}/config.js", timeout=60)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(
        f"{BASE}/auth/authorize_public_link",
        params={"response_type": "code", "client_id": client, "redirect_uri": redirect, "state": state},
        allow_redirects=False,
        timeout=60,
    )
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(
        f"{BASE}/auth/token",
        json={
            "code": query["code"][0], "state": query.get("state", [state])[0],
            "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect,
        },
        timeout=60,
    )
    token.raise_for_status()
    headers = {"Authorization": "Bearer " + token.json()["access_token"]}
    for path in ("/UTSW-Glioma", "/UTSW-Glioma/BT0001", "/UTSW-Glioma/BT0008"):
        reply = session.post(
            f"{BASE}/api/v5/packages/1265/files/received/page",
            headers=headers,
            json={"path": path, "filters": {}},
            timeout=60,
        )
        reply.raise_for_status()
        body = reply.json()
        print(json.dumps({
            "path": path,
            "keys": sorted(body),
            "non_items": {key: value for key, value in body.items() if key != "items"},
            "item_count": len(body.get("items", [])),
            "items": [
                {key: item.get(key) for key in ("path", "basename", "type", "size")}
                for item in body.get("items", [])[:15]
            ],
        }, indent=2))
        token = reply.headers.get("x-aspera-next-iteration-token")
        if path == "/UTSW-Glioma":
            if not token:
                raise RuntimeError("missing next-page iteration token")
            second = session.post(
                f"{BASE}/api/v5/packages/1265/files/received/page",
                headers=headers,
                params={"iteration_token": token},
                json={"path": path, "filters": {}},
                timeout=60,
            )
            second.raise_for_status()
            second_body = second.json()
            print(json.dumps({
                "path": path,
                "page": 2,
                "item_count": len(second_body.get("items", [])),
                "first_path": second_body["items"][0]["path"],
                "last_path": second_body["items"][-1]["path"],
                "has_next_token": bool(second.headers.get("x-aspera-next-iteration-token")),
            }, indent=2))
    requested = [
        "/UTSW-Glioma/BT0001/brain_t1ce.nii.gz",
        "/UTSW-Glioma/BT0002/brain_t1ce.nii.gz",
    ]
    spec_reply = session.post(
        f"{BASE}/api/v5/packages/1265/transfer_spec/download",
        headers=headers,
        params={"transfer_type": "connect", "type": "received"},
        json={"paths": [{"path": path, "type": "file"} for path in requested]},
        timeout=60,
    )
    spec_reply.raise_for_status()
    spec = spec_reply.json()
    print(json.dumps({
        "transfer_spec_keys": sorted(spec),
        "path_rows": [
            {key: row.get(key) for key in ("source", "destination", "source_root", "destination_root")}
            for row in spec.get("paths", [])
        ],
        "top_level_roots": {key: spec.get(key) for key in ("source_root", "destination_root")},
    }, indent=2))


if __name__ == "__main__":
    main()
