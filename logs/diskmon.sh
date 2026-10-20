#!/bin/bash
# 1 s samples: epoch, then per NVMe drive: sectors read, ms busy (io_ticks)
while true; do
  l="$(date +%s.%N)"
  for d in nvme0n1 nvme1n1; do
    l="$l $d $(awk -v d=$d '$3==d{print $6, $13}' /proc/diskstats)"
  done
  echo "$l"
  sleep 1
done
