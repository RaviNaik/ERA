#!/usr/bin/env bash
# Run the whole task inside a Docker container with GPU access.
#
#   ./run_docker.sh                 build the image if needed, then run all four steps (resumable)
#   ./run_docker.sh gpu-test        check that the container sees your GPUs
#   ./run_docker.sh step 3          run only step 1, 2, 3 or 4 (scripts/0N_*.sh)
#   ./run_docker.sh shell           open a shell in the container (same mounts)
#
# Mounts (everything that matters stays on the host):
#   HOST_WORK_DIR (default ./work)  -> /work   models, GGUFs, llama.cpp build, venvs, caches   (~330 GB peak)
#   this folder                     -> /app    config.sh and results/ (scores, runs_row.csv, logs)
# The container runs as your user, so no file on the host is owned by root.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_WORK_DIR="${HOST_WORK_DIR:-$HERE/work}"
IMAGE="${IMAGE:-g12-quant-rotation:cuda12.4}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
GPUS="${GPUS:-all}"                       # e.g. GPUS='"device=0,1"' to pick cards
mkdir -p "$HOST_WORK_DIR/home" "$HERE/results"

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
if [ "${1:-}" != "build-only" ] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" --build-arg "BASE_IMAGE=$BASE_IMAGE" -f "$HERE/docker/Dockerfile" "$HERE/docker"
fi
[ "${1:-}" = "build-only" ] && { docker build -t "$IMAGE" --build-arg "BASE_IMAGE=$BASE_IMAGE" -f "$HERE/docker/Dockerfile" "$HERE/docker"; exit 0; }

case "${1:-all}" in
  gpu-test) CMD=(nvidia-smi) ;;
  step)     n="${2:?usage: ./run_docker.sh step 1|2|3|4}"; CMD=(bash -c "exec scripts/0${n}_*.sh") ;;
  shell)    CMD=(bash) ;;
  all|"")   CMD=(./run_all.sh) ;;
  *)        CMD=("$@") ;;
esac

TTY=(); [ -t 0 ] && [ -t 1 ] && TTY=(-it)
exec docker run --rm ${TTY[@]+"${TTY[@]}"} --gpus "$GPUS" --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$HOST_WORK_DIR:/work" \
  -v "$HERE:/app" \
  -e WORK_DIR=/work -e HOME=/work/home \
  "$IMAGE" "${CMD[@]}"
