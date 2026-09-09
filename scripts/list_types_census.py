import struct, sys
from collections import Counter

p = sys.argv[1]
with open(p, 'rb') as f:
    f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))

    def read_u64():
        return struct.unpack('<Q', f.read(8))[0]

    def read_str():
        n = read_u64()
        return f.read(n).decode('utf-8', errors='replace')

    def skip_value():
        typ = struct.unpack('<I', f.read(4))[0]
        if typ in (0, 1, 7):
            f.read(1)
        elif typ in (2, 3):
            f.read(2)
        elif typ in (4, 5, 6):
            f.read(4)
        elif typ in (10, 11, 12):
            f.read(8)
        elif typ == 8:
            n = read_u64()
            f.read(n)
        elif typ == 9:
            atyp, alen = struct.unpack('<IQ', f.read(12))
            if atyp == 8:
                for _ in range(alen):
                    n = read_u64()
                    f.read(n)
            elif atyp == 9:
                for _ in range(alen):
                    skip_value()
            else:
                size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(atyp, 1)
                f.read(size * alen)
        else:
            raise ValueError(f'unknown kv type {typ}')

    for _ in range(n_kv):
        read_str()
        skip_value()

    counts = Counter()
    examples = {}
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [read_u64() for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = read_u64()
        counts[ttype] += 1
        examples.setdefault(ttype, name)

    for t, cnt in sorted(counts.items()):
        print(t, cnt, examples[t])
