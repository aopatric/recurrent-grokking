#!/usr/bin/env bash
# run training suites on a rented gpu machine over ssh, and stream the results back.
#
#   scripts/vast.sh up     HOST PORT [OPTS] ARGS...   push, smoke test, launch, then watch until done
#   scripts/vast.sh push   HOST PORT                  ship the committed code and the data, install the environment
#   scripts/vast.sh smoke  HOST PORT ARGS...          a few steps of each distinct config (src/smoke.py ARGS)
#   scripts/vast.sh launch HOST PORT ARGS...          start the runner in the background (src/suite.py ARGS)
#   scripts/vast.sh watch  HOST PORT [OPTS]           stream runs/ back every few seconds until nothing is running
#   scripts/vast.sh status HOST PORT                  live jobs, the runner log, gpu use and verdicts per config
#   scripts/vast.sh pull   HOST PORT [--into DIR]     copy runs/ back once, checkpoints included
#   scripts/vast.sh stop   HOST PORT                  pause every running job at its next step
#   scripts/vast.sh shell  HOST PORT                  open a shell in the remote checkout
#
# OPTS: --into DIR (default runs/vast), --every SECONDS (default 60), --no-smoke (up only),
#       --table N (up and watch: print the verdict table of the copied runs every N rounds and at the end).
# ARGS for up and launch are src/suite.py arguments, e.g. --suite tier1 tier2 --until 20000.
# env: VAST_USER (default root), VAST_DIR (default /workspace), ALLOW_DIRTY=1 to ship HEAD despite uncommitted
# changes to tracked files.
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
[ $# -ge 3 ] || usage
cmd=$1 host=$2 port=$3
shift 3

# options shared by up, watch and pull; everything else is passed on
into=runs/vast every=60 smoke=1 table=0 rest=()
while [ $# -gt 0 ]; do
  case $1 in
    --into) into=$2; shift 2 ;;
    --every) every=$2; shift 2 ;;
    --no-smoke) smoke=0; shift ;;
    --table) table=$2; shift 2 ;;
    *) rest+=("$1"); shift ;;
  esac
done
set -- ${rest[@]+"${rest[@]}"}

user=${VAST_USER:-root}
base=${VAST_DIR:-/workspace}
repo="$base/time-loops"
here=$(cd "$(dirname "$0")/.." && pwd)
ssh_opts=(-p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-vast-%C" -o ControlPersist=10m)
target="$user@$host"
# the shared-connection socket lives here
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"

remote() { ssh "${ssh_opts[@]}" "$target" "export PATH=\$HOME/.local/bin:\$PATH; set -euo pipefail; $1"; }
sync_up() { rsync -az --info=stats1 -e "ssh ${ssh_opts[*]}" "$@"; }
quote() { printf '%q ' "$@"; }

# the verdict table of the runs copied so far, computed locally
show_table() {
  local py="$here/.venv/bin/python"
  [ -x "$py" ] || py="uv run python"
  echo "== verdicts so far (in-progress runs are judged at their latest eval; see last_step)"
  $py "$here/src/verdicts.py" --all --root "$into" || true
}

# copy the instance's runs/ into $into. files are rewritten in place, so a tensorboard reading them keeps
# seeing new events; checkpoints are left for the final copy unless $1 is "full"
fetch() {
  local skip=(--exclude suite.pid --exclude train.pid --exclude '*.tmp')
  [ "${1:-}" = full ] || skip+=(--exclude '*.pt')
  mkdir -p "$into"
  rsync -az --inplace "${skip[@]}" ${2:+"$2"} -e "ssh ${ssh_opts[*]}" "$target:$repo/runs/" "$into/"
}

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
          else git -c advice.detachedHead=false clone -q $base/time-loops.bundle $repo; fi
          git -C $repo -c advice.detachedHead=false checkout -q --detach $sha
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
watch)
  echo "== streaming $target:$repo/runs into $into every ${every}s"
  echo "   view live with: uv run tensorboard --logdir $into"
  echo "   ctrl-c stops the streaming only; training keeps going (watch again to pick it back up)"
  idle=0 round=0
  while :; do
    round=$((round + 1))
    fetch || echo "   (copy failed; retrying next round)"
    state=$(remote "cd $repo && { .venv/bin/python src/suite.py --running || true; }
                    echo '~'; tail -n 1 runs/suite.log 2>/dev/null || true" 2>/dev/null) || state="unreachable"
    live=$(sed '/^~$/,$d' <<<"$state" | grep '^training:' || true)
    last=$(sed '1,/^~$/d' <<<"$state")
    echo "$(date '+%H:%M:%S')  $(grep -c . <<<"$live" || true) training  |  runner: ${last:-$state}"
    [ -z "$live" ] || sed 's/^training: /    /' <<<"$live"
    if [ "$table" -gt 0 ] && [ $((round % table)) -eq 0 ]; then show_table; fi
    # two idle rounds in a row, so a runner that is still starting up isn't mistaken for a finished one
    if grep -q '^nothing running' <<<"$state"; then idle=$((idle + 1)); else idle=0; fi
    [ "$idle" -ge 2 ] && break
    sleep "$every"
  done
  echo "== nothing is running; final copy, checkpoints included"
  fetch full
  [ "$table" -gt 0 ] && show_table
  echo "== results in $into"
  echo "   uv run python src/verdicts.py --all --by-config --root $into"
  ;;
up)
  [ $# -ge 1 ] || { echo "error: give src/suite.py arguments, e.g. --suite tier1 --until 20000" >&2; exit 2; }
  "$0" push "$host" "$port"
  if [ "$smoke" = 1 ]; then
    # the smoke test takes the run selection only, not the launch options
    picks=()
    while [ $# -gt 0 ]; do
      case $1 in
        --until|--devices) shift 2 ;;
        --dry-run) shift ;;
        *) picks+=("$1"); shift ;;
      esac
    done
    set -- ${rest[@]+"${rest[@]}"}
    echo "== smoke test"
    "$0" smoke "$host" "$port" ${picks[@]+"${picks[@]}"} ||
      { echo "error: a config failed its smoke test; fix it, commit, and run up again" >&2; exit 1; }
  fi
  echo "== launching"
  "$0" launch "$host" "$port" "$@"
  "$0" watch "$host" "$port" --into "$into" --every "$every" --table "$table"
  ;;
status)
  remote "cd $repo
          .venv/bin/python src/suite.py --running || true
          tail -n 15 runs/suite.log 2>/dev/null || true
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
          .venv/bin/python src/verdicts.py --all --by-config 2>/dev/null || echo 'no metrics yet'"
  ;;
pull)
  fetch full --info=stats1
  echo "== results in $into"
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
