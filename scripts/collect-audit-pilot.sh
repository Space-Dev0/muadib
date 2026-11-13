#!/usr/bin/env bash
# Wait up to six hours for the existing Windows generation task, then package.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
host='myalt@100.78.202.97'
source_dir='outputs/audit-pilot-800000-seeds-20000-20074'
pack_dir='outputs/audit-pilot-packs-800000'
checkpoint='outputs/archived-checkpoints/velocity-online-v1-800000-79c331f.pt'
if [[ -e "$source_dir" || -e "$pack_dir" ]]; then
  echo 'Source or pack directory already exists; inspect before restarting.' >&2
  exit 1
fi
for ((attempt=1; attempt<=360; attempt++)); do
  state=$(ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=15 "$host" \
    'powershell -NoProfile -Command "$n=(Get-ChildItem $env:USERPROFILE\muadib\source\outputs\audit-pilot-800000-seeds-20000-20074 -Filter *.png -ErrorAction SilentlyContinue | Measure-Object).Count; $t=Get-ScheduledTask -TaskName '\''Muadib Pilot Audit Generation 800000'\'' -ErrorAction Stop; $r=(Get-ScheduledTaskInfo -TaskName $t.TaskName).LastTaskResult; if ($n -eq 300 -and $t.State -eq '\''Ready'\'' -and $r -eq 0) { '\''complete'\'' } else { '\''waiting'\'' }"' 2>/dev/null || true)
  state=${state//$'\r'/}
  echo "$(date -u +%FT%TZ) ${state:-unreachable}"
  if [[ "$state" == 'complete' ]]; then
    mkdir "$source_dir"
    scp -F /dev/null -o BatchMode=yes -o ConnectTimeout=15 \
      "$host:muadib/source/outputs/audit-pilot-800000-seeds-20000-20074/*.png" "$source_dir/"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/muadib-uv-cache}" uv run --no-sync \
      python -m exposure.audit_pilot_pack --source "$source_dir" --checkpoint "$checkpoint" --output "$pack_dir"
    echo "Reviewer archives are ready in $pack_dir"
    exit 0
  fi
  sleep 60
done
echo 'Collection timed out; existing GPU images were preserved.' >&2
exit 1
