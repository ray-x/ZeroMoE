# Benchmark profiles and results

This directory contains benchmark-only TOML profiles and the consolidated
performance report. These profiles are tied to specific hardware, model
quantization, cache budgets, prompts, and run settings; they are not general
model defaults.

Use the portable starter profiles in the parent `configs/` directory for
initial setup. See [`perf-baselines.md`](perf-baselines.md) for measurements,
benchmark commands, and the profile used for each recorded result. Detailed
model recipes remain in [`docs/models/`](../../docs/models/).

The six tracked TOMLs are the profiles used by `bench/regression_gate.sh`.
Older A/B and historical result profiles remain in the maintainer's local
testing archive; references to those profiles in the report are not available
in a clean public checkout.

Profiles that are no longer part of the published benchmark catalog remain
available locally under the ignored `configs/testing/` or `configs/obsolete/`
directories. Neither directory is included in the public repository.
