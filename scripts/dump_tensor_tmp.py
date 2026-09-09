import struct, sys
p = sys.argv[1]
target_name = sys.argv[2]
n_dump = int(sys.argv[3]) if len(sys.argv) > 3 else 16
with open(p, 'rb') as f:
    magic = f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
    print(f'ver={ver} tensors={n_tensors} kv={n_kv}')
    def read_u64(): return struct.unpack('<Q', f.read(8))[0]
    def read_str(): 
        n = read_u64()
        return f.read(n).decode('utf-8', errors='replace')
    def skip_value():
        typ = struct.unpack('<I', f.read(4))[0]
        if typ in (0,1,7): f.read(1)
        elif typ in (2,3): f.read(2)
        elif typ in (4,5,6): f.read(4)
        elif typ in (10,11,12): f.read(8)
        elif typ == 8:
            n = read_u64(); f.read(n)
        elif typ == 9:
            atyp, alen = struct.unpack('<IQ', f.read(12))
            if atyp == 8:
                for _ in range(alen): n = read_u64(); f.read(n)
            elif atyp == 9:
                for _ in range(alen): skip_value()
            else:
                size = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(atyp,1)
                f.read(size * alen)
        else: raise ValueError(f'kv type {typ}')
    alignment = 32
    for _ in range(n_kv):
        k = read_str()
        typ = struct.unpack('<I', f.read(4))[0]
        if k == 'general.alignment' and typ == 4:
            alignment = struct.unpack('<I', f.read(4))[0]
        else:
            f.seek(-4, 1)
            skip_value()
    tensor_info_end = f.tell()
    # compute data start alignment
    pad = (alignment - (tensor_info_end % alignment)) % alignment
    data_start = tensor_info_end + pad
    target = None
    for i in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [read_u64() for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = read_u64()
        if name == target_name:
            target = (name, dims, ttype, offset)
            break
    if not target:
        print('tensor not found')
        sys.exit(1)
    name, dims, ttype, offset = target
    print(name, dims, 'type=', ttype)
    nelem = 1
    for d in dims: nelem *= d
    f.seek(data_start + offset)
    if ttype == 0:
        vals = struct.unpack('<' + 'f'*nelem, f.read(nelem*4))
    elif ttype == 1:
        vals = [struct.unpack('<e', f.read(2))[0] for _ in range(nelem)]
    else:
        print('unsupported type for dump')
        sys.exit(1)
    print('first', n_dump, 'vals:', vals[:n_dump])
    print('min=', min(vals), 'max=', max(vals), 'mean=', sum(vals)/len(vals))
