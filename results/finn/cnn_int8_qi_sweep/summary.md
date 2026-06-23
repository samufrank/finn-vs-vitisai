# CNN INT8 QI target_fps sweep — synth/impl resource + fit verdict

Generated 2026-06-09T15:19:53Z. Isolated scratch `/tmp/finn_qi_int8_sweep`. Per-build primary records (runme.log, utilization rpt, FINN report JSONs) in `results/finn/cnn_int8_qi_sweep/<label>/`.

| model | fps | verdict | LUT% | BRAM% | DSP% | CARRY8% | binding | elapsed |
|---|---|---|---|---|---|---|---|---|
| tiny | 200 | FIT | 29.5 | 5.3 | 0.6 | 13.7 | LUT | 1557s |
| tiny | 500 | FIT | 29.6 | 5.6 | 0.8 | 13.8 | LUT | 1544s |
| tiny | 3000 | FIT | 35.6 | 6.2 | 3.1 | 19.6 | LUT | 1642s |
| tiny | 5000 | FIT | 39.6 | 3.9 | 5.3 | 31.2 | LUT | 1662s |
| small | 200 | FIT | 34.1 | 9.5 | 0.8 | 14.2 | LUT | 1533s |
| small | 500 | FIT | 34.0 | 6.5 | 1.9 | 14.2 | LUT | 1529s |
| small | 3000 | FIT | 64.5 | 5.3 | 11.1 | 54.9 | LUT | 2305s |
| medium | 500 | FIT | 87.9 | 14.8 | 9.7 | 55.5 | LUT | 2709s |
| deep_3 | 500 | FIT | 42.4 | 68.5 | 3.6 | 20.4 | BRAM | 1662s |
| deep_3 | 1000 | FIT | 87.6 | 13.2 | 9.7 | 55.1 | LUT | 2293s |
