awk 'FNR>=120&& FNR<=129{printf "%s,\n", $0}' $1
