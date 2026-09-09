import struct, sys
p = sys.argv[1]
with open(p, 'rb') as f:
    magic = f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
    print(f'ver={ver} tensors={n_tensors} kv={n_kv}')
    def read_u64():
        return struct.unpack('<Q', f.read(8))[0]
    def read_str():
        n = read_u64()
        return f.read(n).decode('utf-8', errors='replace')
    def read_str_value():
        n = struct.unpack('<I', f.read(4))[0]
        return f.read(n).decode('utf-8', errors='replace')
    def skip_value():
        typ = struct.unpack('<I', f.read(4))[0]
        # GGUF KV types: 0=UINT8,1=INT8,2=UINT16,3=INT16,4=UINT32,5=INT32,6=FLOAT32,7=BOOL,8=STRING,9=ARRAY,10=UINT64,11=INT64,12=FLOAT64
        if typ in (0,1,7):
            f.read(1)
        elif typ in (2,3):
            f.read(2)
        elif typ in (4,5,6):
            f.read(4)
        elif typ in (10,11,12):
            f.read(8)
        elif typ == 8:
            n = read_u64()
            f.read(n)
        elif typ == 9:
            atyp, alen = struct.unpack('<IQ', f.read(12))
            if atyp == 8:
                for _ in range(alen):
                    n = read_u64(); f.read(n)
            elif atyp == 9:
                for _ in range(alen):
                    skip_value()
            else:
                size = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(atyp,1)
                f.read(size * alen)
        else:
            raise ValueError(f'unknown kv type {typ}')
    for _ in range(n_kv):
        read_str()
        skip_value()
    type_names = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',4:'Q5_0',5:'Q5_1',6:'Q8_0',7:'Q8_1',8:'Q2_K',9:'Q3_K',10:'Q4_K',11:'Q5_K',12:'Q6_K',13:'IQ2_XXS',14:'IQ2_XS',15:'IQ3_XXS',16:'IQ3_S',17:'IQ2_S',18:'IQ4_XS',19:'IQ1_S',20:'IQ1_M',21:'IQ2_M',22:'BF16'}
    # GGUF KV types per gguf_loader.hpp: 0=UINT8,1=INT8,2=UINT16,3=INT16,4=UINT32,5=INT32,6=FLOAT32,7=BOOL,8=STRING,9=ARRAY,10=UINT64,11=INT64,12=FLOAT64
    for i in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [read_u64() for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = read_u64()
        if 'shexp' in name or 'ffn_gate_exps' in name or 'ffn_up_exps' in name or 'ffn_down_exps' in name or 'attn_post_norm' in name or 'attn_norm' in name or 'ssm' in name or 'output' in name or 'token_embd' in name or i < 30:
            print(f'{name:70s} type={ttype:3d} ({type_names.get(ttype,"?")}) dims={dims}')
