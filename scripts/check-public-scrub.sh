#!/usr/bin/env bash
set -euo pipefail

# Public-repo privacy guard: keep private operator paths, key paths, host commands,
# and validation-host details out of tracked files.
patterns=(
  '/Users''/[^/]+/'
  '/home/ubuntu/work''/Camelid'
  'Documents''/cert'
  'ssh ''-i'
  '[A-Za-z0-9._-]+@''[0-9]{1,3}([.][0-9]{1,3}){3}'
  '(^|[^0-9])10[.]([0-9]{1,3}[.]){2}[0-9]{1,3}([^0-9]|$)'
  '(^|[^0-9])192[.]168[.][0-9]{1,3}[.][0-9]{1,3}([^0-9]|$)'
  '(^|[^0-9])172[.](1[6-9]|2[0-9]|3[0-1])[.][0-9]{1,3}[.][0-9]{1,3}([^0-9]|$)'
  '54[.]218[.]217[.]232'
  '[.]pem([^A-Za-z0-9_]|$)'
  '[$]HOME/Desktop/Code/backend|/Desktop/Code/backend'
  'StrictHostKeyChecking=accept-new'
  'target/model-promotion-host-[0-9TZ-]+'
)

status=0
for pattern in "${patterns[@]}"; do
  matches=$(git grep -n -I -E "$pattern" -- \
    ':!.git' \
    ':!target' \
    ':!frontend/dist' \
    ':!frontend/node_modules' \
    ':!scripts/check-public-scrub.sh' || true)
  if [[ -n "$matches" ]]; then
    printf 'public scrub guard failed for pattern: %s\n%s\n' "$pattern" "$matches" >&2
    status=1
  fi
done

exit "$status"
