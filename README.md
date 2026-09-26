# Py-E3SM-EAMxx

A line-by-line PyTorch translation of the E3SM EAMxx atmospheric physics suite
(SHOC, P3, cloud fraction, SPA, TMS, RRTMGP, and optionally ZM), validated
against the running EAMxx code on identical inputs.

Goals: a single-column model here, then a batched, GPU-capable, differentiable
physics package for our dycore (tensors `(ncol[, nmember], nlev)`, index 0 at
model top, pressure interfaces from the host).

- Plan, scope and acceptance tests: [docs/plan.md](docs/plan.md)
- What matches the reference so far: [docs/status.md](docs/status.md)
- Building and running the EAMxx reference: [reference/README.md](reference/README.md)

## Setup

```bash
conda env create -p ~/miniconda3/envs/eamxx-torch -f environment.yml
~/miniconda3/envs/eamxx-torch/bin/pip install --no-deps -e .
~/miniconda3/envs/eamxx-torch/bin/python -m pytest
```

Use conda-forge `pytorch`, not pip `torch`: mixing pip torch with conda numpy
loads two OpenMP runtimes and aborts on macOS.

## License

BSD-3-Clause. Translated code keeps the E3SM and RTE+RRTMGP notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
