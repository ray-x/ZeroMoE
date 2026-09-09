import struct, json, sys

p = sys.argv[1]
with open(p, 'rb') as f:
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    meta = json.loads(f.read(hdr_len))
    names = sorted(meta.keys())
    for n in names:
        if n == '__metadata__':
            continue
        info = meta[n]
        print(f'{n:60s} {info["dtype"]:8s} {tuple(info["shape"])}')
