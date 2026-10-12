#!/bin/sh
# First start on an empty volume: train the model and seed the demo database.
set -e
python -m scripts.seed_demo
exec "$@"
