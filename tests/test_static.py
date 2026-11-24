import re

from fastapi.testclient import TestClient

from app.api import api


def test_ui_files_are_revalidated_on_every_load():
    # regression: after the 5.5 model upgrade a browser kept showing the old
    # model names from its cached index.html / app.js
    client = TestClient(api.app)
    for path in ("/", "/app.js"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers.get("cache-control") == "no-cache"


def test_page_links_assets_by_content_hash():
    # a normal reload rechecks only the page; a stale app.js cached before
    # no-cache existed kept overwriting the model labels. A new URL per
    # content forces the fetch.
    client = TestClient(api.app)
    page = client.get("/").text
    script = re.search(r'src="app\.js\?v=([0-9a-f]{8})"', page)
    style = re.search(r'href="style\.css\?v=([0-9a-f]{8})"', page)
    assert script and style
    asset = client.get(f"/app.js?v={script.group(1)}")
    assert asset.status_code == 200 and "modelHaiku" in asset.text


def test_health_answers_for_the_container_healthcheck():
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
