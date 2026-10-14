#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
BASE_PY="${AFLOW_BASE_PYTHON:-$AFLOW_PYTHON_RESOLVED}"
SITE="${AFLOW_B200_SITE:-/tmp/aflow-sammed3d-b200-site}"
LAUNCHER="${AFLOW_B200_PYTHON:-/tmp/aflow-sammed3d-b200-python}"
TORCH_VERSION="${AFLOW_B200_TORCH_VERSION:-2.12.0}"
TORCHVISION_VERSION="${AFLOW_B200_TORCHVISION_VERSION:-0.27.0}"

mkdir -p "$SITE"
"$BASE_PY" -m pip install --no-cache-dir --upgrade --target "$SITE" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$SITE\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$BASE_PY" "\$@"
EOF
chmod +x "$LAUNCHER"

"$LAUNCHER" - <<'PY'
import torch
import monai
import torchio

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.get_device_capability(0) == (10, 0), torch.cuda.get_device_capability(0)
x = torch.ones((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("python_ok", True)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("matmul", float(y[0, 0]))
print("monai", monai.__version__, "torchio", torchio.__version__)
PY

printf 'B200_ENV_READY=%s\n' "$LAUNCHER"
