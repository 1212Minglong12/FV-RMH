#!/usr/bin/env python
from pathlib import Path
import py_compile
import re
import sys

root = Path(__file__).resolve().parents[1]
files = sorted((root / "scripts").rglob("*.py"))
errors = []
for path in files:
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        errors.append(f"Syntax error: {path.relative_to(root)}: {exc}")

sensitive_patterns = {
    "workstation path": re.compile(r"[A-Za-z]:\\\\Users\\\\", re.I),
    "example personal name": re.compile(r"ChenZhifang|HeJinQuan", re.I),
    "credential assignment": re.compile(r"(?i)(password|passwd|api[_-]?key|secret|token)\\s*=\\s*['\\\"][^'\\\"]+['\\\"]"),
}
for path in files:
    text = path.read_text(encoding="utf-8", errors="replace")
    for label, pattern in sensitive_patterns.items():
        if pattern.search(text):
            errors.append(f"Potential {label}: {path.relative_to(root)}")

print(f"Checked {len(files)} Python scripts.")
if errors:
    print("Validation failed:")
    for item in errors:
        print(" -", item)
    sys.exit(1)
print("Syntax and basic sensitive-string checks passed.")
