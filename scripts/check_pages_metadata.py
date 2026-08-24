from pathlib import Path
import sys

html = Path(sys.argv[1]).read_text(encoding="utf-8")
required = [
    '<link rel="canonical" href="https://peterponyu.github.io/iAODE/"',
    'name="robots" content="index, follow"',
    'property="og:url" content="https://peterponyu.github.io/iAODE/"',
    'name="description"',
]
missing = [token for token in required if token not in html]
if missing:
    raise SystemExit("missing metadata: " + ", ".join(missing))
if '/iAODE/frontend/' in html:
    raise SystemExit("local-only workspace canonical leaked into public artifact")
