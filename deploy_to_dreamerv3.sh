#!/bin/bash
# DreamerV3 코드를 ddori/DreamerV3 레포로 배포하는 스크립트
# 사용법: bash deploy_to_dreamerv3.sh
set -e

REPO_URL="https://github.com/ddori/DreamerV3.git"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR=$(mktemp -d)

echo "=== DreamerV3 배포 시작 ==="
echo "Source: ${SCRIPT_DIR}/dreamer/"
echo "Target: ${REPO_URL}"
echo ""

# 1. DreamerV3 레포 clone
echo "[1/5] Cloning ${REPO_URL}..."
git clone "${REPO_URL}" "${TMPDIR}/DreamerV3"
cd "${TMPDIR}/DreamerV3"

# 2. 기존 파일 정리 (README, LICENSE 등은 유지)
echo "[2/5] Preparing target repo..."
find . -maxdepth 1 -not -name '.git' -not -name '.' -not -name 'README.md' -not -name 'LICENSE' -exec rm -rf {} +
rm -rf dreamer/ models/ utils/

# 3. dreamer 패키지 복사
echo "[3/5] Copying dreamer package..."
cp -r "${SCRIPT_DIR}/dreamer" .
cp "${SCRIPT_DIR}/setup.sh" .
cp "${SCRIPT_DIR}/dreamer/requirements.txt" ./requirements.txt
cp "${SCRIPT_DIR}/.gitignore" .

# 4. Commit
echo "[4/5] Committing..."
git add -A
git commit -m "feat: DreamerV3 world model RL agent (PyTorch + DMC)

Full implementation of DreamerV3 (Hafner et al., 2023):
- RSSM with 32x32 discrete categorical latents + unimix
- CNN encoder/decoder for 64x64 image observations
- Symlog two-hot encoding for reward/value predictions
- Actor-Critic with EMA target + percentile return normalization
- KL balancing with free nats
- Episode replay buffer with sequence sampling
- DeepMind Control Suite environment wrapper

Usage: python -m dreamer --env_name walker_walk --device cuda"

# 5. Push
echo "[5/5] Pushing to ${REPO_URL}..."
git push origin main

echo ""
echo "=== 배포 완료! ==="
echo "https://github.com/ddori/DreamerV3"
echo ""
echo "실행 방법:"
echo "  git clone ${REPO_URL} && cd DreamerV3"
echo "  bash setup.sh"
echo "  python -m dreamer --env_name walker_walk --device cuda"

# 정리
rm -rf "${TMPDIR}"
