from pathlib import Path
import sys

html_path = Path(sys.argv[1])
html = html_path.read_text(encoding="utf-8")
required = [
    '<link rel="canonical" href="https://peterponyu.github.io/iAODE/"',
    'name="robots" content="index, follow"',
    'property="og:url" content="https://peterponyu.github.io/iAODE/"',
    'name="description"',
    'Dataset Browser',
    'https://peterponyu.github.io/scportal/datasets/',
    'Continuity Explorer',
    'https://peterponyu.github.io/scportal/explorer/',
]
missing = [token for token in required if token not in html]
if missing:
    raise SystemExit("missing metadata: " + ", ".join(missing))
if '/iAODE/frontend/' in html:
    raise SystemExit("local-only workspace canonical leaked into public artifact")
if any(token in html.lower() for token in ('localhost', '127.0.0.1', 'start_training_ui')):
    raise SystemExit("local runtime details leaked into public artifact")

root = html_path.parent
documents = {
    "robots.txt": [
        "User-agent: *",
        "Allow: /iAODE/",
        "Sitemap: https://peterponyu.github.io/iAODE/sitemap.xml",
    ],
    "sitemap.xml": [
        "<?xml",
        "<urlset",
        "https://peterponyu.github.io/iAODE/",
    ],
}
for name, required_tokens in documents.items():
    document_path = root / name
    if not document_path.is_file():
        raise SystemExit(f"missing deployment-root document: {document_path}")
    contents = document_path.read_text(encoding="utf-8")
    missing = [token for token in required_tokens if token not in contents]
    if missing:
        raise SystemExit(f"invalid {name}: missing " + ", ".join(missing))
