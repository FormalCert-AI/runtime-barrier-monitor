# Anytime-Valid Runtime Monitoring of Learned Barrier Certificates

Experiment code for the paper.

## Setup

```bash
pip install -r requirements.txt
```

The Python scripts in `synthesis/` also require PyTorch (`pip install torch`). The SOS script requires MATLAB,
YALMIP and MOSEK.

## Reproduce the paper

Run from the repository root. `simulate.py` caches simulations in `results/`. Tables and distribution plots reuse
matching cached results, or simulate them if missing. Figure 2 simulates its example trajectories separately.

```bash
python simulate.py                                # all experiments -> results/*.npz
python -m rq1.table1                              # Table 1, valid certificates (RQ1)
python -m rq2.table1                              # Table 1, invalid certificates (RQ2)
python -m rq3.fig2_evidence                       # Figure 2 (RQ3)
python -m practical_application.results           # Section 4.4: retraining and pre-exit alarms
python -m appendix.counterexample_system2_input   # Appendix A: counterexample, exact arithmetic
python -m appendix.counterexample_system3         # Appendix A: counterexample, interval arithmetic
python -m appendix.fig3_alarm_rates               # Figure 3
python -m appendix.fig4_factor_a                  # Figure 4
python -m appendix.fig5_b0                        # Figure 5
python -m appendix.fig6_baselines                 # Table 3 and Figure 6
python -m appendix.fig7_sensitivity               # Figure 7
```

Figures are written to `results/figures/`.

## Models

| File | Model | Training script |
|---|---|---|
| `models/system2_sos.json` | System 2 barrier (SOS) | `synthesis/system2_sos.m` |
| `models/system3.npz` | System 3 policy and barrier | `synthesis/train_system3.py` |
| `models/system2_input_retrained.npz` | System 2+input barrier, retrained | `synthesis/retrain_system2_input.py` |
| `models/system3_retrained.npz` | System 3 policy and barrier, retrained | `synthesis/retrain_system3.py` |

```bash
python -m synthesis.train_system3
python -m synthesis.retrain_system2_input
python -m synthesis.retrain_system3
```

Training scripts write to `results/synthesis/`; copy a file to `models/` to use it. System 2+input retraining updates
the barrier; System 3 retraining updates both policy and barrier. Both use monitored episodes (Algorithm 1), with
one observed or sampled successor per state (M = 1).

## Settings (`core/config.py`)

N = 1000 initial states, R = 1000 runs from each, K_max = 400, alpha = 0.05, seed 2026. Alarm at the first k >= 1 with
M_k >= a^(K_max - k) / alpha; default K_max ln a = 4 (K_max ln a = 0 is a = 1). DAB: gamma = 0.92, U = 100.

Reported trajectory rates concern the sampled initial states and finite evaluation horizon. `p_max` is the maximum
empirical alarm rate over these initial states. The pre-exit alarm rate is conditional on runs that exit within the
horizon and counts only alarms strictly before the first exit; it is undefined if no runs exit.
