# Documentation

- [Main README](../README.md): purpose, installation and a small calculation.
- [Configuration](configuration.md): JSON inputs, units, measurement model and SQLite outputs.
- [Paper reproduction](reproducing_paper_results.md): self-contained Slurm workflow.
- [Verification](verification.md): completed calculations, numerical agreement and limitations.
- [Public numerical tables](../results/paper/): inspectable endpoints and histories.

## Manuscript sources

`main_rev3.tex` is the author manuscript used by the reproduction guide, based
on [arXiv:2610.16032v1](https://arxiv.org/abs/2610.16032v1) with the
excitation-gate order description corrected to match the implementation.
[Version and license details](manuscript-license.md) distinguish the software
MIT license from the manuscript and figure CC BY 4.0 license.
Earlier drafts and submission correspondence are not included.
The scientific figures in `figures/` are required source assets. Bibliographic
references are listed in the manuscript; third-party article PDFs are not
bundled in the public working tree.

Build from the repository root with a TeX installation that provides `latexmk`,
REVTeX and the packages requested by the chosen source:

```bash
sh docs/build.sh main_rev3
```

Output is written to `docs/build/main_rev3/`, which is ignored by Git. The
builder accepts a manuscript basename or a `.tex` filename. It rejects paths
outside the documentation directory. The root VS Code configuration sends
LaTeX Workshop output to the same build directory.

Raw calculation databases and logs belong in `local_runs/`, not in the
manuscript or figure directories. Compact numerical evidence is versioned
separately under `results/paper/`.
