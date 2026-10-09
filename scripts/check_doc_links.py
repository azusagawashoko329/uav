#!/usr/bin/env python3
"""Check repository-local Markdown links after experiment moves."""
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
problems=[]
for md in ROOT.rglob('*.md'):
    if '.git' in md.parts or any(p.startswith('source_snapshot_') for p in md.parts):continue
    text=md.read_text(errors='replace')
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        if target.startswith(('http:','https:','mailto:','#','data:')):continue
        target=target.split('#',1)[0].strip('<>')
        if not target or '<' in target or '>' in target or ' ' in target:continue
        if not (md.parent/target).exists():problems.append(f'{md.relative_to(ROOT)} -> {target}')
if problems:
    print('Broken relative links:')
    print('\n'.join(problems[:100]))
    sys.exit(1)
print('Relative Markdown links: OK')
