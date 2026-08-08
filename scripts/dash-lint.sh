#!/usr/bin/env bash
#
# dash-lint.sh
#
# Fail if any em dash (U+2014) or en dash (U+2013) appears anywhere in the
# repository's text files. These characters are banned per the project style
# rules (code, comments, commits, markdown, LaTeX). We match them by Unicode
# codepoint through a PCRE pattern so this script itself contains no literal
# dash for it to flag.
#
# Exit code 0 means clean, 1 means at least one offending character was found.
set -euo pipefail

PATTERN='[\x{2013}\x{2014}]'

# Operate from the repository root when we are inside a git tree, otherwise the
# current directory. Prefer the git file list so build artifacts and vendored
# third party trees are never scanned.
if git rev-parse --git-dir >/dev/null 2>&1; then
  cd "$(git rev-parse --show-toplevel)"
  mapfile -d '' FILES < <(git ls-files -z)
else
  mapfile -d '' FILES < <(find . -type f -not -path './.git/*' -print0)
fi

hits=0
for f in "${FILES[@]}"; do
  # Skip files grep judges to be binary.
  if grep -Iq . "$f" 2>/dev/null; then
    if grep -nP "$PATTERN" "$f" >/dev/null 2>&1; then
      grep -nP "$PATTERN" "$f" | sed "s|^|$f:|"
      hits=1
    fi
  fi
done

if [ "$hits" -ne 0 ]; then
  echo "dash-lint: found em dash (U+2014) or en dash (U+2013). Replace with ASCII." >&2
  exit 1
fi

echo "dash-lint: clean"
