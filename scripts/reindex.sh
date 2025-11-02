#!/usr/bin/env bash
set -e
DEVICE=${1:-cuda}
python create_db.py --force --device ${DEVICE}
