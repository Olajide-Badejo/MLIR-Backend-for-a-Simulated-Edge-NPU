# An em dash inside the same fenced block

This is the other half of the exemption. The block below is the same kind of
block as the one in md-flag-in-fence-pass.md, and the flag in it is fine, but
the em dash on the next line is not. No command line ever needs one, so the
code block earns no exemption from the unicode rule.

```bash
bash scripts/regression-baseline.sh --check
echo "this line has an em dash — and must be rejected"
```

The linter must report unicode-dash for this file.
