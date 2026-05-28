#!/usr/bin/bash

# Kill CARLA process on GPU 0
pid=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | grep 'CarlaUE4' | awk -F, '{print $1}' | head -n 1)
if [ -n "$pid" ]; then
	kill -9 "$pid"
	echo "CARLA instance on GPU 0 (PID: $pid) killed."
else
	echo "No CARLA instance found on GPU 0."
fi

# Kill any process holding the traffic manager port (8000) or streaming port (2001)
for port in 8000 2001; do
	tm_pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -n 1)
	if [ -n "$tm_pid" ]; then
		kill -9 "$tm_pid" 2>/dev/null && echo "Process on port $port (PID: $tm_pid) killed." || true
	fi
done
