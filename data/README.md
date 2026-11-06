# Data

The recorded workload used in the evaluation is a ten-day contiguous subset of
the 1998 FIFA World Cup web-site access logs (WorldCup98) from the Internet
Traffic Archive. The archives themselves are not in the repository
(`data/raw/` is ignored by git); everything needed to fetch them again and
check that the copy is identical is recorded here and in `provenance.json`.

## WorldCup98

| | |
|---|---|
| Landing page | https://ita.ee.lbl.gov/html/contrib/WorldCup.html |
| Download base | https://ita.ee.lbl.gov/traces/WorldCup/ |
| Retrieved | 2026-11-05, 16:57 UTC (`provenance.json` → `worldcup98.retrieved_utc`) |
| Subset | days 41–50, i.e. 5–14 June 1998, Paris local time; 41 files, 1,592,308,084 bytes compressed |
| Requests in subset | 250,257,178 |
| Tools archive | `WorldCup_tools.tar.gz`, used for the per-file reference statistics |
| Per-file SHA-256 | `provenance.json` → `worldcup98.files[].sha256` |

**Attribution** (the form the ITA asks for, quoted from the landing page):

> M. Arlitt and T. Jin, "1998 World Cup Web Site Access Logs", August 1998.
> Available at http://www.acm.org/sigcomm/ITA/.

**Licence.** The landing page has two separate grants. The *software and
documentation* (the tools archive) may be used, reproduced, modified and
distributed for non-commercial purposes only. The *access logs* may be used
and redistributed freely as long as the Copyright and Disclaimer travel
unmodified, publications must carry the attribution above, and the ITA asks
for a copy of any publication based on the logs. The notice is reproduced
verbatim in `WORLDCUP98_COPYRIGHT.txt`. This repository ships derived series
(5-second request counts and a scaled copy) in `results/traces/`, not the logs,
but the notice is kept next to them anyway. Use here is non-commercial
research.

### Why days 41–50

* **Contiguous.** The forecaster is trained and evaluated on chronological
  splits of one unbroken series; a gap would put a discontinuity into every
  window that straddles it.
* **Contains the regime change the controller is meant for.** Days 41–45 are
  the last pre-tournament days. Day 46 (10 June) is the opening match, and
  days 46–50 are the first group-stage days, with matches on every day. The
  subset therefore holds quiet pre-tournament traffic, a sharp rise in base
  load, and match-time bursts on top of a diurnal cycle.
* **Size.** About 1.6 GB compressed against roughly 12 GB for the full
  archive, which keeps the download and the full decode-and-validate pass
  practical (about 10 s for all 250 M records).
* **Not chosen by looking at controller results.** The subset was fixed before
  any controller was run on it.

By the archive's own weekday rule (day mod 7 = 0 is a Saturday), days 49 and
50 are Saturday 13 and Sunday 14 June. With the 60/20/20 chronological split
they form the test period, and their traffic is lighter than days 47–48 (see
"Two scalings" below).

### Format and validation

Each request is a fixed 20-byte big-endian record:
`uint32 timestamp, clientID, objectID, size; uint8 method, status, type, server`.
Timestamps are seconds since the epoch in GMT; Paris local time during the
tournament was GMT+2.

`experiments/make_real_traces.py` decodes every file in full and compares it
against the reference statistics in `WorldCup_tools.tar.gz`
(`check/wc_dayX_Y.check`, the archive's `checklog` output): total requests,
total bytes (excluding the `0xFFFFFFFF` no-size sentinel, as `checklog` does),
minimum and maximum timestamp, out-of-order count, number of servers, and the
request count of every server. A single mismatch aborts the conversion. All 41
files matched. The decoder was also checked against the 1,000-record sample
log in the tools archive.

**Observed day boundary.** The landing page describes each day as
00:00:00–23:59:59 local time. The files actually run from 00:00:01 to 00:00:00
the next day inclusive: day 41 starts at 00:00:01 on 5 June, day 50 ends at
exactly 00:00:00 on 15 June, and day 51 starts one second later. The bins
follow the files, so every request in the subset is counted once and none is
borrowed from day 40 or day 51.

**Midnight artefact.** At every local midnight the request count drops for
5–40 s (1–8 bins) to between 0.8 % and 29 % of the surrounding ten-minute
mean, then recovers. All 41 bins that fall below 30 % of their ten-minute running mean
lie within one minute of a local midnight, and none lies anywhere else. This
pattern is consistent with log rotation at the day boundary rather than with
user behaviour. Following the brief, the dips are left in place, not
smoothed. One of them, at 00:00 on 14 June, lies in the test period.

### From requests to an arrival rate

* Requests are counted in 5-second bins (the harness control interval),
  starting at 00:00:01 on 5 June: 172,800 bins.
* Rate in requests/second = count / 5. Nothing is smoothed, filtered or
  interpolated; every burst keeps its timing and its height relative to the
  rest of the series. The measured series is `results/traces/worldcup98_raw.npy`.
* **Scale (modelling assumption).** The simulated pool is μ = 40 req/s per
  replica with N_max = 16, about 640 req/s at saturation, while the raw peak is
  2,316.2 req/s. The series is multiplied by one constant so that its peak
  equals 85 % of 640 req/s (option (a) of the revision brief). The factor,
  0.234867, is recorded in `results/traces/worldcup98.json`. The traffic's
  shape is real; its absolute level relative to the pool is a choice.

### Two scalings

The whole-series peak falls on 10 June, inside the training period. Under that
single factor the test period (13–14 June, a weekend) never exceeds 135 req/s,
about 21 % of pool capacity, and at that load no controller is stressed. Two
families are therefore evaluated and both are reported:

| Family | Factor on the measured rate | Peak, whole series | Peak, test period |
|---|---|---|---|
| `worldcup98` | 0.234867 (whole-series peak at 85 % of capacity) | 544 req/s | 135 req/s |
| `worldcup98_testpeak` | 0.234867 × 4.030978 (test-period peak at 85 %) | 2,193 req/s | 544 req/s |

`worldcup98_testpeak` is the same series times one further constant. The
forecaster's inputs are z-scored with training-period statistics, so a
constant factor cancels exactly and the model sees identical inputs under
both families. The harness checks that the derived series is the base series
times the recorded constant before it uses a forecast. No test-period
information reaches the forecaster through the choice of scale. What the
choice does change is which part of the series loads the pool: under
`worldcup98_testpeak` the training period would saturate it, which is
irrelevant because controllers are only ever evaluated on the test period.

### Splits

Chronological and contiguous, 60/20/20 by bins, with the same arithmetic as
`pact.forecast.train.chronological_window_splits`:

| Split | Bins | Dates (Paris) |
|---|---|---|
| train | 0 – 103,679 | 5–10 June (days 41–46) |
| validation | 103,680 – 138,239 | 11–12 June (days 47–48) |
| test | 138,240 – 172,799 | 13–14 June (days 49–50) |

The splits fall on whole days. Normalisation statistics come from the training
split only. Controller experiments on the recorded trace use disjoint 400-tick
windows taken only from the test split, each preceded by at least 48 ticks of
test-split history (the forecaster's input window). There are 86 such
windows.

### Reproduce

```bash
mkdir -p data/raw/worldcup98 && cd data/raw/worldcup98
curl -O https://ita.ee.lbl.gov/traces/WorldCup/WorldCup_tools.tar.gz
for d in $(seq 41 50); do for p in $(seq 1 8); do
  curl -sf -O https://ita.ee.lbl.gov/traces/WorldCup/wc_day${d}_${p}.gz || true
done; done
cd ../../..
python experiments/make_real_traces.py      # validates, bins, scales, writes provenance
```

Compare the SHA-256 of each file with `provenance.json` before relying on the
copy.

## Bitbrains GWA-T-12 (located, verified, not used)

* Landing page: https://atlarge-research.com/gwa-t-12/
* Archives (HTTP 200 on 2026-11-05):
  `https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip`
  (134,047,207 bytes) and `https://atlarge-research.com/gwa-traces/gwa_t_12_rnd.zip`
  (163,099,172 bytes).
* Citation given on the landing page: S. Shen, V. van Beek and A. Iosup,
  "Statistical Characterization of Business-Critical Workloads Hosted in Cloud
  Datacenters", 15th IEEE/ACM International Symposium on Cluster, Cloud and
  Grid Computing (CCGrid), 2015.

It was not used. The trace records per-VM CPU and memory utilisation sampled
every 5 minutes, not request arrivals. Using it to drive a 5-second controller
would need 60× temporal interpolation, which invents every sub-5-minute
dynamic the controller responds to. It would also need an inversion of
utilisation through the capacity model to an implied arrival rate, which
assumes the very model under test. No result in this repository comes from it.
