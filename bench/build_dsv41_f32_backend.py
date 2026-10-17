#!/usr/bin/env python3
"""Build the optional F32 hook in isolation, reusing a native GGML build.

Only ggml-cuda.cu is recompiled. Input source/build trees are read-only. Object
files are snapshotted before linking and released afterward; model files are
never accessed. Use LD_LIBRARY_PATH=<out> with the MoEStream executable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--ggml-source', type=Path, required=True)
p.add_argument('--ggml-build', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
work = out / 'build-work'
work.mkdir(exist_ok=True)
source = a.ggml_source.resolve() / 'src/ggml-cuda/ggml-cuda.cu'
base = a.ggml_build.resolve() / 'src/ggml-cuda'
target = base / 'CMakeFiles/ggml-cuda.dir'
patch = root / 'patches/ggml/0010-native-f32-quant-graph-hook.patch'
patched = work / 'ggml-cuda.cu'
applied = subprocess.run(['patch', '--batch', '--dry-run', '--reverse',
                          '--output', '/dev/null', str(source), str(patch)],
                         capture_output=True).returncode == 0
if applied:
    # The source tree already carries the hook; reuse it unchanged.
    shutil.copyfile(source, patched)
else:
    subprocess.run(['patch', '--batch', '--forward', '--output', str(patched),
                    str(source), str(patch)], check=True)
flags = {}
for line in (target / 'flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
cache = (a.ggml_build.resolve().parent / 'CMakeCache.txt').read_text()
nvcc = next((line.split('=',1)[1] for line in cache.splitlines()
             if line.startswith('CMAKE_CUDA_COMPILER:FILEPATH=')), '/opt/cuda/bin/nvcc')
host = next((line.split('=',1)[1] for line in cache.splitlines()
             if line.startswith('CMAKE_CUDA_HOST_COMPILER:FILEPATH=')), None)
command = [nvcc]
if host:
    command += ['-ccbin', host]
command += flags['CUDA_DEFINES'] + flags['CUDA_INCLUDES'] + flags['CUDA_FLAGS']
command += ['--allow-unsupported-compiler', '-c', str(patched), '-o', str(work/'ggml-cuda.cu.o')]
print('Compiling isolated CUDA dispatcher', flush=True)
subprocess.run(command, cwd=base, check=True)

# Take a stable snapshot of every object used by the link. Other sessions may
# rebuild the reference worktree while this experiment is running.
objects = shlex.split((target / 'objects1.rsp').read_text())
snapshot = work / 'objects'
snapshot.mkdir(exist_ok=True)
paths = []
for index, item in enumerate(objects):
    src = (base / item).resolve()
    if src.name == 'ggml-cuda.cu.o' and src.parent == target:
        paths.append(work/'ggml-cuda.cu.o')
    else:
        dst = snapshot / (str(index) + '-' + src.name)
        shutil.copyfile(src, dst)
        paths.append(dst)
(work/'objects.rsp').write_text(' '.join(json.dumps(str(x)) for x in paths))
libs = shlex.split((target / 'linkLibs.rsp').read_text())
for i, item in enumerate(libs):
    if item.startswith('-Wl,-rpath,'):
        libs[i] = '-Wl,-rpath,' + str(out)
    elif not item.startswith('-') and (base/item).exists():
        libs[i] = str((base/item).resolve())
(work/'libs.rsp').write_text(' '.join(json.dumps(x) for x in libs))
link = shlex.split((target / 'link.txt').read_text())
for i, item in enumerate(link):
    if item.startswith('@'):
        link[i] = '@' + str(work / ('objects.rsp' if 'objects' in item else 'libs.rsp'))
    elif item.startswith('-Wl,--dependency-file='):
        link[i] = '-Wl,--dependency-file=' + str(work/'link.d')
    elif i and link[i-1] == '-o':
        link[i] = str(out/'libggml-cuda.so.0')
print('Linking isolated CUDA backend', flush=True)
subprocess.run(link, cwd=base, check=True)
shutil.rmtree(snapshot)
for name in ('libggml.so.0', 'libggml-base.so.0', 'libggml-cpu.so.0'):
    shutil.copyfile(a.ggml_build.resolve()/'src'/name, out/name)
for name in ('libggml.so.0','libggml-base.so.0','libggml-cpu.so.0','libggml-cuda.so.0'):
    alias = out/name[:-2]
    if alias.is_symlink():
        alias.unlink()
    if not alias.exists():
        alias.symlink_to(name)
manifest = {
    'source': str(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
    'libraries': {x.name: hashlib.sha256(x.read_bytes()).hexdigest()
                  for x in out.glob('*.so.0')},
}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Backend ready:',out,flush=True)
