#!/usr/bin/env python3
"""Re-embed code/training/make_roc.py (canonical) into kai-roc-r5.yaml's ConfigMap.

Run from the repo root after ANY edit to code/training/make_roc.py:
    python3 code/jobs/training/reembed_make_roc.py

Idempotent: replaces everything between the `make_roc.py: |` key and the `---`
document separator, so it works both on the fresh __MAKE_ROC_PY__ placeholder and
on an already-embedded copy.
"""
import pathlib
import re
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[3]      # repo root
SRC = ROOT / "code" / "training" / "make_roc.py"
YAML = ROOT / "code" / "jobs" / "training" / "kai-roc-r5.yaml"

body = textwrap.indent(SRC.read_text(), "    ")
if not body.endswith("\n"):
    body += "\n"

text = YAML.read_text()
# NB: replacement is a FUNCTION so backslash sequences in the source (e.g. "\n"
# inside f-strings) are inserted literally, not interpreted by re.sub.
new = re.sub(r"(  make_roc\.py: \|\n).*?(?=^---$)",
             lambda m: m.group(1) + body, text, flags=re.S | re.M)
assert "__MAKE_ROC_PY__" not in new, "placeholder still present after embed"
assert new != text or body in text, "nothing replaced — yaml structure changed?"
YAML.write_text(new)
print(f"embedded {SRC.relative_to(ROOT)} ({len(body.splitlines())} lines) into {YAML.relative_to(ROOT)}")
