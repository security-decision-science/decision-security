# Decision-security

Reusable **decision-science utilities for security** — Monte Carlo risk bands, Bayesian updates & calibration, survival helpers, Value of Information, light causal helpers, and visualization.

```bash
pip install decision-security  # (will install once you publish)
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


### `LICENSE` (MIT)
```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
...
(standard MIT text unchanged)