# Round-2 logs from sol1 (collected 2026-10-11)

`r2_logs_sol1.tar.gz` (SHA-256 `03c66372ee8a2ffa4d9eba00e488c45893fcd4bc2afffcc37a3765e56c0e3577`): every round-2 job log
from sol1, kept for provenance.

- `logs/r2_score/*.log`: one per scheduled scoring task, ending with the FLAC decode report (DEV-9 amendment 3).
- `logs/r2_*.log`: the downloads, queues A and B, the W1-finish chain, both schedulers, the E6 silence job, both Pascu
  MLAAD-5 reruns (amendments 4 and 5; the a5 score log shows the two clips repeated to 400 samples) and the W4 chain.
- `r2_runs/logs/*.log`: the eleven training runs' console logs.
- `xfer/r2_events.log` and its state: the run -> verify -> next loop's event history.
- `xfer/mlaad5_sizes.txt`, `xfer/mlaad5_short.txt`: the MLAAD-5 longest and shortest clip scans behind amendments 4 and 5.
