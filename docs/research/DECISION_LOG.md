# Decision log

Append-only causal decisions. Later corrections must be new dated entries.

## 2026-09-27 — Guard-capacity contrast

Why: Protocol diagnosis links accumulation to parameter drift and ranking loss. Oracle diagnosis shows only 0.02243 percentage-point EER gain within guarded EP and zero selector regret. Its guard frequently activates, so guard severity is a plausible, untested cause of weak correction.

What the existing results rule out: On target10, changing only reset policy or choosing a different existing guarded hyperparameter is unlikely to explain a large missing gain. The results do not distinguish an overrestrictive guard from a weak objective or representation.

Next design: Hold the existing view-variance objective, margin regularizer, projection, features, optimizer, and episodic reset fixed. Compare production hard guard, a single relaxed guard threshold, and no guard at three representative K/lr/rho settings. Use per-domain ranking plus per-sample helpful/harmful and source-anchor damage to choose whether EPDC needs softer preservation, a reliability controller, or a new objective.
