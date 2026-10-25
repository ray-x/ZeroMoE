import struct
import sys

p = sys.argv[1] if len(sys.argv) > 1 else None
if not p:
    raise SystemExit(f"usage: {sys.argv[0]} <model.gguf>")
with open(p,'rb') as f:
    print('magic',f.read(4))
    ver=struct.unpack('<I',f.read(4))[0]
    print('version',ver)
    n_tensors,n_kv=struct.unpack('<QQ',f.read(16))
    print('n_kv',n_kv)
    def read_u64(): return struct.unpack('<Q',f.read(8))[0]
    def read_str():
        n=read_u64(); return f.read(n).decode('utf-8',errors='replace')
    def val_str(typ):
        if typ in (0,1,7): return f.read(1).hex()
        if typ in (2,3): return struct.unpack('<h' if typ==2 else '<H',f.read(2))[0]
        if typ in (4,5,6): return struct.unpack('<i' if typ==4 else '<I',f.read(4))[0]
        if typ in (10,11,12): return struct.unpack('<q' if typ==10 else '<Q' if typ==11 else '<d',f.read(8))[0]
        if typ==8:
            n=read_u64(); return f.read(n).decode('utf-8',errors='replace')
        if typ==9:
            atyp,alen=struct.unpack('<IQ',f.read(12))
            if atyp==8:
                return [read_str() for _ in range(alen)]
            if atyp in (0,1,7):
                size=1
                return list(f.read(size*alen))
            if atyp in (2,3):
                return list(struct.unpack('<'+('h' if atyp==2 else 'H')*alen, f.read(2*alen)))
            if atyp in (4,5,6):
                return list(struct.unpack('<'+('i' if atyp==4 else 'I')*alen, f.read(4*alen)))
            if atyp in (10,11,12):
                return list(struct.unpack('<'+('q' if atyp==10 else 'Q' if atyp==11 else 'd')*alen, f.read(8*alen)))
            return f'array(type={atyp},len={alen})'
        return f'?type={typ}'
    for _ in range(n_kv):
        k=read_str()
        typ=struct.unpack('<I',f.read(4))[0]
        v=val_str(typ)
        print(k, typ, v if isinstance(v,str) or not isinstance(v,list) else (v[:5] if len(v)>5 else v))
