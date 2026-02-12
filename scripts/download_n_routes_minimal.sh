#!/usr/bin/env bash
set -euo pipefail

# Download and extract only N routes from the Hugging Face dataset without pulling all LFS data.
#
# Usage:
#   bash scripts/download_n_routes_minimal.sh 10
#
# Run from repository root.

N_ROUTES="${1:-10}"
REPO_ROOT="$(pwd)"
ZIP_REPO_DIR="data/carla_leaderboard2/zip"
HF_DATASET_URL="https://huggingface.co/datasets/ln2697/lead_carla"

echo "[1/6] Prepare zip repo folder"
rm -rf "${ZIP_REPO_DIR}"
git clone --filter=blob:none "${HF_DATASET_URL}" "${ZIP_REPO_DIR}"

echo "[2/6] Select ${N_ROUTES} route zip paths"
cd "${ZIP_REPO_DIR}"
mapfile -t ROUTES < <(git ls-tree -r --name-only HEAD | grep '\.zip$' | shuf -n "${N_ROUTES}")

if [[ "${#ROUTES[@]}" -eq 0 ]]; then
  echo "No route zip files found in dataset repo tree."
  exit 1
fi

echo "Selected routes:"
printf '  %s\n' "${ROUTES[@]}"

echo "[3/6] Download only selected zips"
for r in "${ROUTES[@]}"; do
  mkdir -p "$(dirname "$r")"
  wget -q --show-progress \
    "${HF_DATASET_URL}/resolve/main/${r}" \
    -O "${r}"
done

echo "[4/6] Unzip selected routes"
cd "${REPO_ROOT}"
for r in "${ROUTES[@]}"; do
  unzip -o "${ZIP_REPO_DIR}/${r}" -d .
done

echo "[5/6] Verify route count and integrity"
python - <<'PY'
from pathlib import Path

root = Path("data/carla_leaderboard2/data")
routes = sorted([p for p in root.glob("*/*") if p.is_dir()])
print(f"Found {len(routes)} route dirs under {root}")
ok = 0
for r in routes:
    req = ["rgb", "lidar", "radar", "metas"]
    if all((r / x).exists() for x in req):
        ok += 1
print(f"Routes with rgb/lidar/radar/metas: {ok}/{len(routes)}")
PY

echo "[6/6] Optional cleanup to save space"
# Keep extracted data; remove downloaded zips + git metadata/pointers.
find "${ZIP_REPO_DIR}" -type f -name '*.zip' -delete
rm -rf "${ZIP_REPO_DIR}/.git"

echo "Done. Next step (optional): python scripts/build_cache.py"
