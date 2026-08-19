"""Source files are not .tex, so the double hyphen rule does not reach them.

A Python file that documents a flag writes the flag. The linter leaves it
alone because Python has no dash ligature, and this fixture pins that.
"""

# Run the fast subset with --self-test, and the whole sweep with --all-files.
ARGS = ["--check", "--allow-empty-runs", "-j6"]

# An ASCII hyphen in prose is always fine: out of tree, phase p0-foundations.
