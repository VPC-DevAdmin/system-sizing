#!/bin/bash
# Wall power from the BMC over IPMI (DCMI), for hosts without an
# iDRAC credentials file. Appends "epoch,watts" every INTERVAL seconds.
#   ipmi_power.sh out.csv [interval_s]
out=${1:?out.csv}; iv=${2:-2}
while true; do
  w=$(sudo -n ipmitool dcmi power reading 2>/dev/null | awk '/Instantaneous power reading/ {print $4}')
  [ -n "$w" ] && echo "$(date +%s.%N | cut -c1-14),$w" >> "$out"
  sleep "$iv"
done
