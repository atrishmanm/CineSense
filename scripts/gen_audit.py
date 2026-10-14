"""Generate machine-verified audit items (file:line precise) for docs/AUDIT.md."""
import os
import re

SKIP_DIRS = ('.git', '.venv', '__pycache__', '.pytest_cache')


def files(exts=('.py',)):
    out = []
    for dp, dn, fns in os.walk('.'):
        if any(x in dp for x in SKIP_DIRS):
            continue
        for fn in fns:
            if fn.endswith(exts) and fn != 'gen_audit.py':
                out.append(os.path.join(dp, fn))
    return sorted(out)


def lines(f):
    try:
        return open(f, encoding='utf-8', errors='replace').read().splitlines()
    except OSError:
        return []


items = []  # (category, file, line, text)


def add(cat, f, i, text):
    items.append((cat, f.replace(os.sep, '/'), i, text))


for f in files():
    for i, l in enumerate(lines(f), 1):
        s = l.strip()
        if re.match(r'except\s*:', s):
            add('bare-except', f, i, 'bare `except:` swallows everything incl. KeyboardInterrupt/SystemExit')
        if 'TODO' in s or 'FIXME' in s or s.startswith('# XXX') or 'HACK' in s.upper() and 'HACK' in s:
            add('todo', f, i, s[:110])
        if re.search(r'SELECT\s+\*', s, re.I):
            add('select-star', f, i, '`SELECT *` — breaks on schema change, hides column use')
        if re.search(r'time\.sleep\s*\(', s):
            add('sleep', f, i, 'blocking sleep (retry/backoff without jitter, or test slowdown)')
        if re.search(r'localhost|127\.0\.0\.1', s) and 'test' not in f and 'localhost:{' not in s:
            add('localhost', f, i, 'hardcoded localhost URL')
        m = re.search(r'request\.(args\.get|form\.get)\(', s)
        if m:
            add('unvalidated-input', f, i, f'untrusted `{m.group(0)}` — check for caps/validation')
        if re.search(r'request\.(json|get_json|data|values)\b', s):
            add('unvalidated-input', f, i, 'untrusted request body — check for None/type validation')
        if re.match(r'return\s+(\[\]|\{\}|None)\s*(#|$)', s) and '/ai/' in f.replace(os.sep, '/'):
            add('empty-fallback', f, i, 'empty return in ai/ — may hide failure (verify logged)')
        if re.search(r'\b1500\b', s) and 'test' not in f:
            add('magic-number', f, i, 'hardcoded 1500 (ELO start — should be Config.INITIAL_ELO_SCORE)')
        if re.search(r'\b0\.8932\b', s):
            add('stale-claim', f, i, 'removed 0.8932 RMSE claim still present')

# prints in library code (ai/api/app/training/inference, not tests/benchmarks/scripts)
for f in files():
    r = f.replace(os.sep, '/')
    if not r.startswith(('ai/', 'api/', 'app', 'training/', 'inference/', 'tmdb/', 'database/')):
        continue
    if 'test' in r or '_archive' in r:
        continue
    for i, l in enumerate(lines(f), 1):
        if re.search(r'(?<![\w.])print\s*\(', l):
            add('print-in-library', f, i, 'print() in library code — use logging')

cats = {}
for cat, f, i, text in items:
    cats.setdefault(cat, []).append((f, i, text))

n = 0
with open('scripts/audit_items.txt', 'w', encoding='utf-8') as out:
    for cat in sorted(cats):
        out.write(f'## {cat} ({len(cats[cat])})\n')
        for f, i, text in cats[cat]:
            n += 1
            out.write(f'{n}. {f}:{i} — {text}\n')
        out.write('\n')
print('machine items:', n, {k: len(v) for k, v in cats.items()})
