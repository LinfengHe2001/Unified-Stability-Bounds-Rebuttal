import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
secs = re.split(r'^## (.+)$', txt, flags=re.M)[1:]
for name, body in zip(secs[::2], secs[1::2]):
    b = body.strip()
    ph = re.findall(r'\{[^}]*\}', b)
    print(f"{name:32s} chars={len(b):5d}  placeholders={len(ph):2d} ({sum(map(len,ph))} chars)  urls={bool(re.search(r'https?://|www\\.', b))}")
