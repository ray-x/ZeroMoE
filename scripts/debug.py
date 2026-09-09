import struct, sys
p = sys.argv[1]
with open(p, 'rb') as f:
    magic = f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
    print(f'magic={magic} version={ver} tensors={n_tensors} kv={n_kv}')

    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8')

    def read_value(typ):
        if typ in (0,1):   # uint8/int8
            return struct.unpack('<b' if typ else '<B', f.read(1))[0]
        if typ in (2,3):   # uint16/int16
            return struct.unpack('<h' if typ==3 else '<H', f.read(2))[0]
        if typ in (4,5):   # uint32/int32
            return struct.unpack('<i' if typ==5 else '<I', f.read(4))[0]
        if typ in (6,7):   # uint64/int64
            return struct.unpack('<q' if typ==7 else '<Q', f.read(8))[0]
        if typ == 8:       # float32
            return struct.unpack('<f', f.read(4))[0]
        if typ == 9:       # float64
            return struct.unpack('<d', f.read(8))[0]
        if typ == 10:      # bool
            return f.read(1) != b'\x00'
        if typ == 11:      # string
            return read_str()
        if typ == 12:      # array
            atyp, alen = struct.unpack('<IQ', f.read(12))
            arr = []
            for _ in range(alen):
                arr.append(read_value(atyp))
            return arr
        raise ValueError(f'unknown kv type {typ}')

    # Skip KV metadata
    for _ in range(n_kv):
        read_str()              # key
        read_value(struct.unpack('<I', f.read(4))[0])

    # Read tensor info
    for i in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        print(f'{name:60s} type={ttype:3d} dims={dims} offset={offset}')
