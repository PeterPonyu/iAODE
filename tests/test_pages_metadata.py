from pathlib import Path


def test_public_page_contains_scpportal_tool_routes() -> None:
    html = (Path(__file__).parents[1] / "site" / "index.html").read_text(encoding="utf-8")
    assert "Dataset Browser" in html
    assert "https://peterponyu.github.io/scportal/datasets/" in html
    assert "Continuity Explorer" in html
    assert "https://peterponyu.github.io/scportal/explorer/" in html
    assert "localhost" not in html.lower()
    assert "/iAODE/frontend/" not in html
