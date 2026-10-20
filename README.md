# Anytime-Valid Runtime Monitoring of Learned Barrier Certificates

This repository releases the code for the paper, to support reproduction of its experiments. The code is directly
runnable after installing the packages in `requirements.txt` (`pip install -r requirements.txt`). Run every command
from the repository root.

## Repo Structure

- [`simulate.py`](simulate.py): runs all experiments once and stores the results in `results/`
  (`python simulate.py`). Every script below reads these results.
- [`rq1`](rq1), [`rq2`](rq2), [`rq3`](rq3): the research questions of Section 4.
  - [`rq1/table1.py`](rq1/table1.py): Table 1, valid certificates.
  - [`rq2/table1.py`](rq2/table1.py): Table 1, invalid certificates.
  - [`rq3/fig2_evidence.py`](rq3/fig2_evidence.py): Figure 2.
- [`practical_application`](practical_application): Section 4.4.
  - [`results.py`](practical_application/results.py): alarm rates before and after retraining, and alarms before the exit.
- [`appendix`](appendix): Appendices A and B.
  - [`counterexample_system2_input.py`](appendix/counterexample_system2_input.py),
    [`counterexample_system3.py`](appendix/counterexample_system3.py): the counterexamples of Appendix A.
  - [`fig3_alarm_rates.py`](appendix/fig3_alarm_rates.py), [`fig4_factor_a.py`](appendix/fig4_factor_a.py),
    [`fig5_b0.py`](appendix/fig5_b0.py), [`fig6_baselines.py`](appendix/fig6_baselines.py) (also Table 3),
    [`fig7_sensitivity.py`](appendix/fig7_sensitivity.py): Figures 3 to 7.
- [`synthesis`](synthesis): how the models were obtained.
  - [`system2_sos.m`](synthesis/system2_sos.m): SOS certificate of System 2 (MATLAB, YALMIP, MOSEK).
  - [`train_system3.py`](synthesis/train_system3.py): policy and barrier of System 3.
  - [`retrain_system2_input.py`](synthesis/retrain_system2_input.py),
    [`retrain_system3.py`](synthesis/retrain_system3.py): retraining from monitored episodes (Section 4.4).
- [`models`](models): the certificates, barriers and policies used in the paper.
- [`core`](core): systems, monitors, simulation and metrics shared by all scripts.
- [`plotting`](plotting): figure style.

Scripts are run as modules, e.g. `python -m rq1.table1`. Tables are printed and figures are written to
`results/figures/`. For the results, please refer to the paper.

## Contributions

We welcome pull requests. Please describe what the change does and why; for larger changes, please open an issue
first.

## Getting in Touch

For questions, bug reports or feature requests, please open an issue.

## Citation

The citation will be added after publication.
