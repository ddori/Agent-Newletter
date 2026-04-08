#!/bin/bash
# DreamerV3 setup script
# Usage: bash setup.sh

set -e

echo "=== Installing DreamerV3 dependencies ==="

# System dependencies for dm-control (MuJoCo rendering)
# Ubuntu/Debian:
if command -v apt-get &> /dev/null; then
    echo "Installing system packages (requires sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libgl1-mesa-glx libosmesa6-dev libglfw3 patchelf
fi

# Python dependencies
pip install -r dreamer/requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run training:"
echo "  python -m dreamer --env_name walker_walk --device cuda"
echo ""
echo "Quick test (CPU, small model):"
echo "  python -m dreamer --env_name walker_walk --device cpu --total_steps 1000 --prefill_steps 500 --eval_every 500 --log_every 100"
