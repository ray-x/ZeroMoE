import struct
p='/Users/rayxu/.lmstudio/models/Qwen/Qwen3.6-35B-A3B-UD/Qwen3.6-35B-A3B-UD-IQ2_M.gguf'
with open(p,'rb') as f:
    f.read(4); f.read(4); n_tensors,n_kv=struct.unpack('<QQ',f.read(16))
    def read_u64(): return struct.unpack('<Q',f.read(8))[0]
    def read_str():
        n=read_u64(); return f.read(n)
    alignment=32
    for _ in range(n_kv):
        k=read_str().decode('utf-8',errors='replace')
        typ=struct.unpack('<I',f.read(4))[0]
        if k=='general.alignment' and typ==4:
            alignment=struct.unpack('<I',f.read(4))[0]
        elif typ in (0,1,7): f.read(1)
        elif typ in (2,3): f.read(2)
        elif typ in (4,5,6): f.read(4)
        elif typ in (10,11,12): f.read(8)
        elif typ==8: n=read_u64(); f.read(n)
        elif typ==9:
            atyp,alen=struct.unpack('<IQ',f.read(12))
            if atyp==8:
                for _ in range(alen): n=read_u64(); f.read(n)
            elif atyp==9:
                for _ in range(alen):
                    t=struct.unpack('<I',f.read(4))[0]
                    if t==8: n=read_u64(); f.read(n)
                    else: raise
            else:
                size={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(atyp,1)
                f.read(size*alen)
    tensor_info_end=f.tell()
    pad=(alignment-(tensor_info_end%alignment))%alignment
    data_start=tensor_info_end+pad
    # find output_norm
    target=None
    for _ in range(n_tensors):
        name=read_str().decode('utf-8',errors='replace')
        n_dims=struct.unpack('<I',f.read(4))[0]
        dims=[read_u64() for _ in range(n_dims)]
        ttype=struct.unpack('<I',f.read(4))[0]
        offset=read_u64()
        if name=='output_norm.weight':
            target=(name,dims,ttype,offset); break
    print('data_start',data_start,'target',target)
    f.seek(data_start+target[3])
    raw=f.read(64)
    print('hex', raw.hex())
    floats=struct.unpack('<16f', raw)
    print('floats',floats)
