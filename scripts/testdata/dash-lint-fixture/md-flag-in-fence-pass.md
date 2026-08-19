# A flag inside a fenced block

This half of the exemption says a command line flag survives. Rewriting the
flag would make the document wrong, which is the whole reason the exemption
exists.

```bash
bash scripts/regression-baseline.sh --check
ninja -C build -j6
./build/bin/nbin_decode_fuzzer -max_total_time=60 fuzz/corpus
```

A backticked span carries a flag too: `--allow-empty-runs` and `--self-test`.

Outside any block, a double hyphen is still only two hyphens in markdown, so
this line is clean as well.
