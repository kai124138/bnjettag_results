# mulder — the Vitis HLS synthesis box

My working reference for the machine that does all the C-synthesis. NRP Nautilus has no
Xilinx backend, so hls4ml's `build` stage — the only stage that produces real
LUT/FF/DSP/BRAM and latency-in-cycles — runs here and nowhere else. The workflow itself
(what to ship, how to parse the reports, where the numbers go) is in
the synthesis runbook; this file is just the machine.

**Host:** `mulder.t2.ucsd.edu` — a UCSD Tier-2 box, shared with the group.

## Access

`~/.ssh/config` has both the FQDN and a short alias, so `ssh mulder` is enough:

```
Host mulder
    HostName mulder.t2.ucsd.edu
    User kayamaguchi
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Home is NFS-mounted and shared with the rest of the Tier-2 (44 TB, and it does run
close to full — check `df -h ~` before shipping a large project).

## Toolchain

- **Vitis HLS 2023.2**, installed at `/data/software/xilinx/Vitis/2023.2/`.
- The **full** `settings64.sh` must be sourced, not just the HLS one — it provides both
  `vitis-run` and `vitis_hls`. `bnjettag/code/hgq2/mulder_csynth.sh` does this already;
  the reason is written up in `bnjettag/code/hls/RUN_CSYNTH_ON_VITIS.md` (2026-07-02
  correction).
- Target part `xcvu13p-flga2577-2-e` (VU13P), 2.5 ns clock. csynth needs no license;
  place-and-route would.
- OS is AlmaLinux 8.10, so this is not the same environment as the Ubuntu 22.04 that
  `home-pc-cluster-recreation.md` assumes.

## Hardware

Measured 2026-07-24 with `ssh mulder 'lscpu; free -g'`:

| | |
| --- | --- |
| CPU | 2 × AMD EPYC 7302 (Zen 2, "Rome") |
| Cores / threads | 32 physical / 64 logical (16C/32T per socket, 2 NUMA nodes) |
| Clocks | 3.0 GHz base, 3.3 GHz boost (observed 3.29–3.30 GHz on loaded cores) |
| Cache | 16 MB L3 per CCX, 256 MB total across both sockets |
| RAM | 125 GB, plus 7 GB swap |

## What that means in practice

**Vitis HLS C-synthesis is single-threaded.** A synthesis pegs one core at 100% and
leaves the other 63 idle — on 2026-07-24 the box was carrying a single `vitis_hls`
process at 13 days elapsed, 12.5 GB RSS, with a load average of 1.00. The clang-LTO
non-convergence documented in `bnjettag/results/hgq2/constraints_map.md` is the same
observation from the failure side: `clang -cc1` at 100% CPU for three hours without ever
reaching a report.

So mulder is not a fast machine per synthesis — those EPYC cores are a 2019 server part
at 3.3 GHz, and any modern desktop core would beat them roughly two-to-one on a single
run. What mulder gives us is **capacity**: 125 GB of RAM, and enough threads to keep
several syntheses in flight at once.

Which makes memory, not CPU, the thing that actually bites. Three concurrent syntheses
OOM-killed the `w1a4_rf8` run at 54 GB RSS (rc=137, 2026-07-18). The retry went behind a
guard that waits for **≤2 vitis processes AND ≥40 GB free** before starting, under
`setsid` so it survives the session — that guard is the concurrency policy, and it is
worth reusing rather than reinventing. The 6.4M-parameter monolith needs >100 GB on its
own, which is why it only ever runs here.

## Gotchas

- **Check for other people's jobs before launching.** `ps -eo pcpu,rss,etime,comm --sort=-pcpu | head`
  tells you what is already running and how much memory it holds. It is a shared box.
- **A long-running synthesis is not necessarily a hung one** — the attention core has
  sat in "global binding" for 3.5 h and finished. But past a day or so, suspect the LTO
  non-convergence in `constraints_map.md` rather than waiting it out.
- **Nothing on mulder is backed up by us.** Bring the raw `csynth.xml` and the parsed
  JSON home into the store that owns the run (`bnjettag/results/final/` or
  `bnjettag/r7/results/csynth/`) as soon as a run lands.
- **No browser here**, so kubectl/OIDC needs the device-code flow — see
  `nrp-nautilus-setup.md`.
