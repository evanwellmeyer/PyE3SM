#!/bin/bash
# Reproduce the reference source tree: E3SM and scmlib at the pinned commits,
# the submodules EAMxx needs, and this project's patches.
#   bash reference/scripts/setup_reference.sh
# Then: fetch_inputdata.sh, configure_eamxx.sh, and build (see reference/README.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REF_ROOT="$(cd "$HERE/.." && pwd)"

E3SM_SHA=80fe0a71cfabd0091eaacecab7a1ce991563a6d7    # E3SM master, 2026-09-18
SCMLIB_SHA=2dc3f1073a5d03b5f32617e79d68fb20a63cfb43  # scmlib master, 2026-09-16

fetch_at () {  # dir url sha
  if [[ -d "$1/.git" ]]; then return; fi
  git init -q "$1"
  git -C "$1" remote add origin "$2"
  git -C "$1" fetch -q --depth 1 origin "$3"
  git -C "$1" checkout -q FETCH_HEAD
}

fetch_at "$REF_ROOT/E3SM"  https://github.com/E3SM-Project/E3SM.git  "$E3SM_SHA"
fetch_at "$REF_ROOT/scmlib" https://github.com/E3SM-Project/scmlib.git "$SCMLIB_SHA"

cd "$REF_ROOT/E3SM"
GH='url.https://github.com/.insteadOf=git@github.com:'
git -c "$GH" submodule update --init --depth 1 \
  externals/ekat externals/scorpio externals/mam4xx externals/YAKL cime \
  components/eam/src/physics/rrtmgp/external \
  components/eam/src/physics/cosp2/external
git -C externals/ekat -c "$GH" submodule update --init --depth 1 extern/kokkos
git -C cime -c "$GH" submodule update --init --depth 1 CIME/non_py/cprnc

for p in "$REF_ROOT"/patches/*.patch; do
  if git apply --check "$p" 2>/dev/null; then
    git apply "$p"; echo "applied $(basename "$p")"
  else
    echo "skip $(basename "$p") (already applied?)"
  fi
done
