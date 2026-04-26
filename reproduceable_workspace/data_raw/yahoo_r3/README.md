# Yahoo! R3 Raw Data Placeholder

Place the official Yahoo! R3 files here before running the smoke test:

```text
ydata-ymusic-rating-study-v1_0-train.txt
ydata-ymusic-rating-study-v1_0-test.txt
```

The smoke-test script samples 20% of observed rows from each split and refuses to
use RL4Rec pseudo-ground-truth or model-output matrices as a replacement for the
raw observational train split.
