import struct,array,sys
def load(p):
    d=open(p,'rb').read(); o=0; r={}
    while o<len(d):
        s,n=struct.unpack_from('II',d,o); o+=8
        a=array.array('f'); a.frombytes(d[o:o+4*n]); o+=4*n; r[s]=a
    return r
ref=load(sys.argv[1])
for name in sys.argv[2:]:
    c=load(name); rows=[]
    for s in sorted(set(ref)&set(c)):
        x,y=ref[s],c[s]
        d=max(abs(p-q) for p,q in zip(x,y)); i1=max(range(len(x)),key=x.__getitem__)
        rows.append((s,d,sum(1 for q in y if q>y[i1])))
    ds=sorted(r[1] for r in rows)
    print(name,"top1 kept %d/%d"%(sum(1 for r in rows if r[2]==0),len(rows)),"step0 dmax %.3f"%rows[0][1],"median dmax %.3f max %.3f"%(ds[len(ds)//2],ds[-1]))
