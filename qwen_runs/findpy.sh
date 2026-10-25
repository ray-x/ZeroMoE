for p in /home/ray/Projects/ZeroMoE/api/venv/bin/python3 /home/ray/vllm-workspace/venv/bin/python3 /home/ray/.venv/bin/python3 /usr/bin/python3 /usr/bin/python3.14 $(ls -d /home/ray/*/bin/python3 /home/ray/*/*/bin/python3 /home/ray/.local/share/*/bin/python3 2>/dev/null); do
  [ -x "$p" ] && echo "$p: $($p -c 'import numpy,torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
done
