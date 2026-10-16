#!/usr/bin/env bash
# run training suites on a rented gpu machine over ssh, and copy the results back.
#
#   scripts/vast.sh push   HOST PORT            ship the committed code and the data, install the environment
#   scripts/vast.sh smoke  HOST PORT ARGS...    a few steps of each distinct config (src/smoke.py ARGS)
#   scripts/vast.sh launch HOST PORT ARGS...    start the runner in the background (src/suite.py ARGS)
#   scripts/vast.sh status HOST PORT            runner state, gpu use and the verdict table
#   scripts/vast.sh pull   HOST PORT            copy runs/ back into the local runs/; safe to repeat
#   scripts/vast.sh stop   HOST PORT            pause every running job at its next step
#   scripts/vast.sh shell  HOST PORT            open a shell in the remote checkout
#
# env: VAST_USER (default root), VAST_DIR (default /workspace), ALLOW_DIRTY=1 to ship HEAD despite uncommitted
# changes to tracked files.
set -euo pipefail

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 3 ] || usage
cmd=$1 host=$2 port=$3
shift 3

user=${VAST_USER:-root}
base=${VAST_DIR:-/workspace}
repo="$base/time-loops"
here=$(cd "$(dirname "$0")/.." && pwd)
ssh_opts=(-p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-vast-%r@%h:%p" -o ControlPersist=10m)
target="$user@$host"
# the shared-connection socket lives here
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"

remote() { ssh "${ssh_opts[@]}" "$target" "export PATH=\$HOME/.local/bin:\$PATH; set -euo pipefail; $1"; }
sync_up() { rsync -az --info=stats1 -e "ssh ${ssh_opts[*]}" "$@"; }
quote() { printf '%q ' "$@"; }
case $cmd in
push)
  cd "$here"
  if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
    echo "error: uncommitted changes would not be shipped; commit them, or set ALLOW_DIRTY=1 to ship HEAD" >&2
    git status --short --untracked-files=no >&2
    exit 1
  fi
  untracked=$(git status --porcelain | grep -c '^??' || true)
  [ "$untracked" = 0 ] || echo "note: $untracked untracked files are not shipped (git status shows them)"
  sha=$(git rev-parse HEAD)
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  git bundle create "$tmp/time-loops.bundle" HEAD
  (cd data/synthetic && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$tmp/data.sha256"

  echo "== preparing $target"
  remote "mkdir -p $base
          if [ -x $repo/.venv/bin/python ] && (cd $repo && .venv/bin/python src/suite.py --running >/dev/null 2>&1); then
            echo 'error: training is running on the instance; stop it before replacing the code' >&2; exit 1
          fi
          missing=''
          for tool in rsync git curl; do command -v \$tool >/dev/null || missing=\"\$missing \$tool\"; done
          [ -z \"\$missing\" ] || (apt-get update -qq && apt-get install -y -qq \$missing)
          command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"

  echo "== shipping commit ${sha:0:7} and the data"
  sync_up "$tmp/time-loops.bundle" "$tmp/data.sha256" "$target:$base/"
  remote "if [ -d $repo/.git ]; then git -C $repo fetch -q $base/time-loops.bundle HEAD
          else git clone -q $base/time-loops.bundle $repo; fi
          git -C $repo checkout -q --detach $sha
          mkdir -p $repo/data/synthetic $repo/runs"
  sync_up data/synthetic/ "$target:$repo/data/synthetic/"
  remote "cd $repo/data/synthetic && sha256sum --quiet -c $base/data.sha256 && echo 'data checksums match'"

  echo "== installing the environment"
  remote "cd $repo && uv sync --frozen --no-dev -q && .venv/bin/python -c '
import torch
ok = torch.cuda.is_available()
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpus\", torch.cuda.device_count() if ok else 0)
if not ok:
    raise SystemExit(\"error: torch sees no gpu; the instance driver may be older than this torch build needs\")
'"
  echo "== ready: $repo at ${sha:0:7}"
  ;;
smoke)
  remote "cd $repo && .venv/bin/python src/smoke.py $(quote "$@")"
  ;;
launch)
  [ $# -ge 1 ] || { echo "error: give src/suite.py arguments, e.g. --suite tier1 --until 20000" >&2; exit 2; }
  remote "cd $repo
          if .venv/bin/python src/suite.py --running; then echo 'error: stop the running jobs first' >&2; exit 1; fi
          .venv/bin/python src/suite.py $(quote "$@") --dry-run
          setsid nohup .venv/bin/python src/suite.py $(quote "$@") >> runs/suite.log 2>&1 < /dev/null &
          echo \"runner started (process \$!); output in $repo/runs/suite.log\""
  ;;
status)
  remote "cd $repo
          .venv/bin/python src/suite.py --running || true
          tail -n 15 runs/suite.log 2>/dev/null || true
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
          .venv/bin/python src/verdicts.py --all --by-config 2>/dev/null || echo 'no metrics yet'"
  ;;
pull)
  mkdir -p "$here/runs"
  rsync -az --info=stats1 --exclude suite.pid --exclude train.pid --exclude '*.tmp' -e "ssh ${ssh_opts[*]}" "$target:$repo/runs/" "$here/runs/"
  ;;
stop)
  remote "cd $repo && { .venv/bin/python src/suite.py --stop || true; }"
  ;;
shell)
  ssh -t "${ssh_opts[@]}" "$target" "cd $repo && exec \$SHELL -l"
  ;;
*)
  usage
  ;;
esac
