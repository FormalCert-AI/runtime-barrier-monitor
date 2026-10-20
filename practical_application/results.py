"""Section 4.4: alarm rates before/after retraining and pre-exit alarms on the deployed System 3."""
from core.experiments import run
from core.report import p_max, pre_exit_alarm_rate, rates, rho, stay_rate, table

rows = []
for name, label in (('system2_input', 'System 2+input'), ('system3', 'System 3')):
    for suffix, stage in (('', 'deployed'), ('_retrained', 'retrained')):
        results = run(name + suffix)
        d, o = rates(results, 'direct'), rates(results, 'dab')
        rows.append([f'{label}, {stage}', stay_rate(results), rho(d), p_max(d), rho(o), p_max(o)])
print(table(['system', 'in S through K_max', 'Direct rho', 'Direct p_max', 'DAB rho', 'DAB p_max'], rows))

results = run('system3')
print(f'\nDeployed System 3: {100 - stay_rate(results):.2f}% of sampled runs exit by K_max; '
      'pre-exit alarms among exiting runs:')
for c in (4.0, 0.0):
    print(f'  K_max ln a = {c:g}:  DAB {pre_exit_alarm_rate(results, "dab", c):.2f}%   '
          f'Direct {pre_exit_alarm_rate(results, "direct", c):.2f}%')
