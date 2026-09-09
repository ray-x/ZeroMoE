#!/usr/bin/env python3
import struct, sys

p = sys.argv[1] if len(sys.argv) > 1 else ""
if not p:
    print("Usage: python3 debug_list_tensors.py <model.gguf>")
    sys.exit(1)

with open(p, 'rb') as f:
    magic = f.read(4)
    if magic != b'GGUF':
        print(f"Not a GGUF file: {magic}")
        sys.exit(1)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
    print(f'magic={magic} version={ver} tensors={n_tensors} kv={n_kv}')

    def read_u64():
        return struct.unpack('<Q', f.read(8))[0]

    def read_str():
        n = read_u64()
        if n > 100000:
            raise ValueError(f"Suspicious string length {n} at offset {f.tell()}")
        return f.read(n).decode('utf-8')

    def skip_value():
        typ = struct.unpack('<I', f.read(4))[0]
        if typ in (0,1):
            f.read(1)
        elif typ in (2,3):
            f.read(2)
        elif typ in (4,5):
            f.read(4)
        elif typ in (6,7):
            f.read(8)
        elif typ == 8:
            f.read(4)
        elif typ == 9:
            f.read(8)
        elif typ == 10:
            f.read(1)
        elif typ == 11:
            read_str()
        elif typ == 12:
            atyp, alen = struct.unpack('<IQ', f.read(12))
            size = 1
            if atyp in (0,1): size = 1
            elif atyp in (2,3): size = 2
            elif atyp in (4,5): size = 4
            elif atyp in (6,7): size = 8
            elif atyp == 8: size = 4
            elif atyp == 9: size = 8
            elif atyp == 10: size = 1
            elif atyp == 11:
                for _ in range(alen):
                    read_str()
                return
            else:
                raise ValueError(f"unknown array eltype {atyp}")
            f.read(size * alen)
        else:
            raise ValueError(f"unknown kv type {typ}")

    # Skip KV metadata
    for _ in range(n_kv):
        read_str()  # key
        skip_value()

    # Read tensor info
    for i in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [read_u64() for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = read_u64()
        print(f'{name:60s} type={ttype:3d} dims={dims} offset={offset}')
