[![PyPI](https://img.shields.io/pypi/v/decision-security?label=PyPI&include_prereleases)](https://pypi.org/project/decision-security/)
[![Python versions](https://img.shields.io/pypi/pyversions/decision-security.svg)](https://pypi.org/project/decision-security/)
[![CI](https://github.com/security-decision-science/decision-security/actions/workflows/ci.yml/badge.svg)](https://github.com/security-decision-science/decision-security/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/voiculaura/)](https://www.linkedin.com/in/voiculaura/)


# Decision Security

Reusable **decision-science utilities for security** — Monte Carlo risk bands, Bayesian updates & calibration, survival helpers, Value of Information, light causal helpers, and visualization.

## Install
Pre-release for now:
```bash
pip install --pre decision-security
# or pin:
# pip install decision-security==0.1.0a9 
```

## Quickstart

```python
import numpy as np
from decision_security.montecarlo import risk_bands, var_es, make_lognormal_severity, simulate_aggregate_losses

sev = make_lognormal_severity(meanlog=8.0, sdlog=1.2)
losses = simulate_aggregate_losses(n_periods=10000, lam=0.6, severity_sampler=sev)
print(risk_bands(losses))      # {'p50': ..., 'p90': ..., 'p95': ...}
print(var_es(losses))          # (VaR95, ES95)
```

## Modules
	•	synth: synthetic data (heavy-tail losses, counts, mixtures, survival with censoring, categorical/Dirichlet).
	•	montecarlo: Poisson frequency + severity, risk bands, VaR/ES.
	•	bayes: Beta-Binomial & Normal(known σ) updates, calibration helpers.
	•	survival: simple Kaplan–Meier & Nelson–Aalen estimates.
	•	voi: Expected Value of Perfect Information (EVPI) and simple ROI selection.
	•	causal: tiny DAG utilities (parents, descendants, naive backdoor set).
	•	viz: small matplotlib helpers (loss distribution, risk bands, KM curves).

Status: 0.x (APIs may change). MIT License.

See docs & examples: Security Decision Science Book and the Security Decision Labs playground.

## Contributing

Issues and PRs welcome. For non-public questions, contact me on LinkedIn.
