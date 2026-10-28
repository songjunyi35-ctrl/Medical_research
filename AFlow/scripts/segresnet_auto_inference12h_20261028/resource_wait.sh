#!/bin/sh
cd /mlx_devbox/users/songjunyi.666/playground/AFlow || exit 1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /mlx_devbox/users/songjunyi.666/playground/sammed3d-venv/bin/python -m scripts.segresnet_auto_inference12h_20261028.resource_wait
