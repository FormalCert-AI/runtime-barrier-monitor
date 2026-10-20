# Anytime-Valid Runtime Monitoring of Learned Barrier Certificates

Code for the paper *Anytime-Valid Runtime Monitoring of Learned Barrier Certificates*.

Run commands from the repository root with Python 3.12:

```sh
pip install -r requirements.txt
```

## Experiments

```sh
python simulate.py
```

This computes the six experiments in `core/experiments.py` and saves `results/<experiment>.npz`.
Existing caches are reused when their recorded settings match. Individual experiments can be selected with
`python simulate.py system1 system3`.

The following commands use these caches. If a cache is missing or its recorded settings differ, the command
computes that experiment before producing the output.

| Output | Command |
| --- | --- |
| Table 1, valid certificates | `python -m rq1.table1` |
| Table 1, invalid certificates | `python -m rq2.table1` |
| Section 4.4 statistics | `python -m practical_application.results` |
| Figure 3 | `python -m appendix.fig3_alarm_rates` |
| Figure 4 | `python -m appendix.fig4_factor_a` |
| Figure 5 | `python -m appendix.fig5_b0` |
| Table 3 and Figure 6 | `python -m appendix.fig6_baselines` |
| Figure 7 | `python -m appendix.fig7_sensitivity` |

Tables are printed to the terminal; figures are saved in `results/figures/`.

`python -m rq3.fig2_evidence` simulates separate example trajectories for Figure 2.
`python -m appendix.counterexample_system2_input` and `python -m appendix.counterexample_system3`
compute the counterexamples in Appendix A directly from the model files.

## Models

Evaluation uses the files in `models/`:

| File | Experiment |
| --- | --- |
| `system2_sos.json` | System 2 and System 2+input: SOS barrier |
| `system3.npz` | System 3: original policy and barrier |
| `system2_input_retrained.npz` | System 2+input: retrained barrier |
| `system3_retrained.npz` | System 3: retrained policy and barrier |

System 1 uses `B(x) = x^2` directly.

`python -m synthesis.train_system3`, `python -m synthesis.retrain_system2_input`, and
`python -m synthesis.retrain_system3` run training and write weights to `results/synthesis/`.
To evaluate newly trained weights, copy them to the corresponding file in `models/` and remove the affected
experiment cache. `synthesis/system2_sos.m` writes the SOS barrier to `results/synthesis/system2_sos.json`;
it requires MATLAB, YALMIP, and MOSEK, with their paths set in the script.

## Settings

Defaults in `core/config.py`: 1,000 initial states, 1,000 runs per state, `K_MAX=400`, `ALPHA=0.05`,
`K_MAX ln a=4`, `GAMMA=0.92`, `CAP=100`, and seed `2026`. The horizon sweep uses `a=1`.
Alarm rates count alarms at or before the earlier of the first exit and the horizon. Pre-exit alarm rates count alarms
strictly before the first exit, among runs that exit within the horizon.

## Citation

```bibtex
@unpublished{cao2026anytime,
  title  = {Anytime-Valid Runtime Monitoring of Learned Barrier Certificates},
  author = {Cao, Zhipeng and Wu, Taoran and Cui, Weiru and Ong, Luke and Xue, Bai},
  year   = {2026},
  note   = {Manuscript}
}
```
