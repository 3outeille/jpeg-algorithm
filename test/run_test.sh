#!/bin/bash

pytest test_compression.py -s
echo "[test_compression]: OK"
pytest test_decompression.py -s
echo "[test_decompression]: OK"
pytest test_io.py -s
echo "[test_io]: OK"

