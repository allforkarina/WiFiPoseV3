#!/bin/bash

echo "===== $(date) start ====="

python tools/run_linux_formal.py --track non_dann --variant accuracy
python tools/run_linux_formal.py --track non_dann --variant balanced
python tools/run_linux_formal.py --track non_dann --variant diversity

python tools/run_linux_formal.py --track dann --variant accuracy
python tools/run_linux_formal.py --track dann --variant balanced
python tools/run_linux_formal.py --track dann --variant diversity

echo "===== $(date) done ====="
