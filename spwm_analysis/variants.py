"""Worst-case length per response: for every '[A] ... [B] ...' pair keep the longer
branch, for '{a | b}' / '{a/b}' keep the longer option, and fill numeric {X} with 6 chars."""
import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
secs = re.split(r'^## (.+)$', txt, flags=re.M)[1:]
def worst(b):
    # [A] x [B] y   (within one paragraph)
    def branches(line):
        parts = re.split(r'\[[A-C]\] ', line)
        if len(parts) < 3:
            return line
        return parts[0] + max(parts[1:], key=len)
    b = '\n'.join(branches(l) for l in b.split('\n'))
    def opt(m):
        inner = m.group(1)
        parts = re.split(r'\s\|\s|/', inner) if ('|' in inner or '/' in inner) else [inner]
        if re.fullmatch(r'[A-Za-z]?|X|CI|N|R|V|n|m|\d+', inner.strip()):
            return 'x' * (12 if inner == 'CI' else 6)
        if inner.strip() in ('runs', 'list'):
            return 'x' * 40
        return max(parts, key=len)
    for _ in range(3):
        b = re.sub(r'\{([^{}]*)\}', opt, b)
    return b
for name, body in zip(secs[::2], secs[1::2]):
    w = worst(body.strip())
    print(f"{name:32s} worst-case chars={len(w):5d}  {'OK' if len(w) <= 1500 else 'OVER'}")
