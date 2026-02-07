#!/usr/bin/env bash
# Run tests inside Docker container matching CI environment
#
# Usage:
#   scripts/test-docker.sh              # Tier 1: unit + guardrail (~90s)
#   scripts/test-docker.sh --all        # All tiers with data mounts
#   scripts/test-docker.sh --tier 2     # Tier 2: data tests
#   scripts/test-docker.sh --data       # Tier 1+2 with data mounts
#   scripts/test-docker.sh -- -k "test_registry"  # Pass extra pytest args
#
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="foundation-plr-test:latest"

TIER="1"
MOUNT_DATA=false
EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)  TIER="all"; MOUNT_DATA=true; shift ;;
        --tier) TIER="$2"; shift 2 ;;
        --data) MOUNT_DATA=true; TIER="data"; shift ;;
        --)     shift; EXTRA="$*"; break ;;
        *)      EXTRA="$*"; break ;;
    esac
done

echo "Building test image..."
DOCKER_BUILDKIT=1 docker build -f "$PROJECT_ROOT/Dockerfile.test" -t "$IMAGE" "$PROJECT_ROOT"

VOLS=(
    -v "$PROJECT_ROOT/src:/project/src:ro"
    -v "$PROJECT_ROOT/tests:/project/tests:ro"
    -v "$PROJECT_ROOT/configs:/project/configs:ro"
    -v "$PROJECT_ROOT/scripts:/project/scripts:ro"
)

if $MOUNT_DATA; then
    [[ -d "$PROJECT_ROOT/data/r_data" ]]  && VOLS+=(-v "$PROJECT_ROOT/data/r_data:/project/data/r_data:ro")
    [[ -d "$PROJECT_ROOT/data/public" ]]  && VOLS+=(-v "$PROJECT_ROOT/data/public:/project/data/public:ro")
fi

IGNORE="--ignore=tests/test_docker_r.py --ignore=tests/test_docker_full.py"

case $TIER in
    1)    MARK="-m 'unit or guardrail'" ;;
    2)    MARK="-m data" ;;
    3)    MARK="-m 'integration or e2e'" ;;
    data) MARK="-m 'unit or guardrail or data'" ;;
    all)  MARK="" ;;
    *)    MARK="" ;;
esac

echo "Running Tier $TIER tests in Docker..."
docker run --rm -e MPLBACKEND=Agg "${VOLS[@]}" "$IMAGE" \
    bash -c "python -m pytest tests/ $MARK $IGNORE -n auto -v --tb=short $EXTRA"
