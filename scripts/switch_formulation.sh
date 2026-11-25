#!/usr/bin/env bash
# Switch margin_kind of the controller: edits config.yaml, regenerates the acados code, rebuilds.
# Restart sim_full.launch.py afterwards.
#
#   ./switch_formulation.sh leonardelli | support | min_max
set -e
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$PKG_DIR/config/config.yaml"

case "$1" in
  leonardelli|support|min_max) ;;
  *) echo "usage: $0 leonardelli|support|min_max"; exit 1 ;;
esac

sed -i -E "s/^(  margin_kind: )\"[a-z_]+\"/\1\"$1\"/" "$CFG"
echo "config.yaml:"; grep -E "^  (margin_kind|safety_factor):" "$CFG"

cd "$PKG_DIR/scripts"
LD_LIBRARY_PATH=$HOME/acados/lib PYTHONPATH= ~/opti_pessi_acados/acados_env/bin/python model_gen.py > /tmp/model_gen.log 2>&1 \
  || { echo "model_gen.py failed, see /tmp/model_gen.log"; exit 1; }
echo "acados code regenerated"

# MAKEFLAGS=-j2: the generated code is big, with more jobs the build runs out of memory (6 GB)
cd ~/colcon_ws
source install/setup.bash
MAKEFLAGS=-j2 colcon build --symlink-install --packages-select acados_legged_controller --parallel-workers 1 \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo 2>&1 | grep -E "error|Finished|Failed" || true
echo "done: restart sim_full.launch.py"
