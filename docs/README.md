# Research documentation

- [Branch overview and installation](../README.md).
- [Analytic Newton OO and shared Pauli RDM evaluation](newton_oo.md).
- [RDM reconstruction and uncertainty-aware OO](uncertainty_oo.md).
- [Shot-noise evaluation and scoring](lih_shot_noise_evaluation.md).
- [Manuscript version and CC BY 4.0 license](manuscript-license.md).
- [Public history and numerical provenance](history-rewrite.md).

`main_rev3.tex` is the author manuscript based on
[arXiv:2610.16032v1](https://arxiv.org/abs/2610.16032v1), with the excitation-gate
order description corrected. The required scientific figures are in `figures/`.
Earlier drafts, submission correspondence and downloaded article PDFs are not
included. With REVTeX, the required packages and `latexmk`, build from the root:

```bash
sh docs/build.sh main_rev3
```

Generated files go to ignored `docs/build/main_rev3/`. Use the
[main branch](https://github.com/QS-laboratory/oouccd-rs/tree/main) for the
paper baseline's verified Slurm workflow and compact numerical tables.
