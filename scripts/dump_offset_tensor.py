import sys
sys.path.insert(0, "/Users/rayxu/github/llama.cpp/gguf-py")
from gguf.gguf_reader import GGUFReader

path = sys.argv[1]
r = GGUFReader(path)

rows = []
for t in r.tensors:
    rows.append((t.data_offset, t.n_bytes, t.name))

rows.sort(key=lambda x: x[0])

prev_end = None
for off, nbytes, name in rows:
    flag = ""
    if prev_end is not None:
        if off < prev_end:
            flag = "  <<< OVERLAP with previous tensor's end (%d)!" % prev_end
        elif off > prev_end:
            gap = off - prev_end
            if gap > 4096:
                flag = "  (gap of %d bytes before this tensor)" % gap

    marker = ""
    if name == "blk.0.ffn_gate_tid2eid.weight":
        marker = "  <<<<<< TID2EID"

    boundary_note = ""
    for b, label in [(2**31, "2^31"), (2**32, "2^32")]:
        if off < b <= off + nbytes:
            boundary_note = boundary_note + ("  [CROSSES %s]" % label)

    line = "offset=%14d end=%14d n_bytes=%12d %s%s%s%s" % (
        off, off + nbytes, nbytes, name, marker, flag, boundary_note
    )
    print(line)
