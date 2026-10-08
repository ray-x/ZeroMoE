"""Minimal GGUF v3 reader: tensor name -> (dims, ggml_type, absolute file offset). Works with quant types the gguf package does not know."""
import struct

_SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

def _read_str(f):
    (n,) = struct.unpack('<Q', f.read(8)); return f.read(n).decode('utf8', 'replace')

def _read_val(f, t):
    if t in _SCALAR:
        fmt = '<' + _SCALAR[t]; return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8: return _read_str(f)
    if t == 9:
        (et,) = struct.unpack('<I', f.read(4)); (n,) = struct.unpack('<Q', f.read(8))
        if et in _SCALAR:
            sz = struct.calcsize('<' + _SCALAR[et]); f.seek(sz * n, 1); return None
        for _ in range(n): _read_val(f, et)
        return None
    raise ValueError('bad kv type %d' % t)

def parse(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        (ver,) = struct.unpack('<I', f.read(4)); nt, nkv = struct.unpack('<QQ', f.read(16))
        align = 32
        for _ in range(nkv):
            k = _read_str(f); (t,) = struct.unpack('<I', f.read(4)); v = _read_val(f, t)
            if k == 'general.alignment' and v: align = v
        infos = {}
        for _ in range(nt):
            name = _read_str(f); (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd)); (ty,) = struct.unpack('<I', f.read(4)); (off,) = struct.unpack('<Q', f.read(8))
            infos[name] = (dims, ty, off)
        base = (f.tell() + align - 1) // align * align
    return {n: (d, t, base + o) for n, (d, t, o) in infos.items()}

def read_f32(path, entry):
    import numpy as np
    dims, ty, off = entry; n = 1
    for d in dims: n *= d
    if ty == 0: return np.fromfile(path, dtype=np.float32, count=n, offset=off)
    if ty == 1: return np.fromfile(path, dtype=np.float16, count=n, offset=off).astype(np.float32)
    if ty == 30:
        u = np.fromfile(path, dtype=np.uint16, count=n, offset=off).astype(np.uint32) << 16
        return u.view(np.float32)
    raise ValueError('tensor type %d is not float' % ty)
