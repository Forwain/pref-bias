#!/bin/bash
# Copy the code (not data) to CARC: ~/pref-bias. Run from Git Bash in this folder.
set -e
cd "$(dirname "$0")"
tar czf - --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache . | ssh carc 'mkdir -p ~/pref-bias && tar xzf - -C ~/pref-bias'
echo "synced to carc:~/pref-bias"
