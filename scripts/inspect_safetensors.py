import struct, json, sys
from collections import Counter

p = sys.argv[1]
with open(p, 'rb') as f:
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    print(f'header_len={hdr_len}')
    hdr = f.read(hdr_len)
    try:
        meta = json.loads(hdr)
    except Exception as e:
        print('JSON parse failed:', e)
        print(hdr[:2000])
        sys.exit(1)

    dtypes = Counter()
    shapes = []
    for name, info in list(meta.items())[:50]:
        if name == '__metadata__':
            continue
        dtypes[info['dtype']] += 1
        shapes.append((name, info['dtype'], tuple(info['shape'])))

    print('dtypes:', dict(dtypes))
    print('first 30 tensors:')
    for name, dt, sh in shapes[:30]:
        print(f'  {name:60s} {dt:8s} {sh}')

    if '__metadata__' in meta:
        print('__metadata__:', json.dumps(meta['__metadata__'], indent=2)[:2000])
