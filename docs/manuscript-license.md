# Manuscript version and license

The manuscript `main_rev3.tex` and all scientific figure assets in `figures/`
are by Hiromichi Hayashi and Takeshi Sato, copyright 2026, and are provided
under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The software MIT license does not replace that license.

Original publication: Hiromichi Hayashi and Takeshi Sato,
*Sequential optimization for orbital-optimized variational quantum eigensolvers*,
[arXiv:2610.16032v1](https://arxiv.org/abs/2610.16032v1), 12 October 2026.
The arXiv record identifies the preprint's CC BY 4.0 license.

The included author source corrects the excitation-gate order description
relative to arXiv v1: the implementation applies double-excitation gates
before single-excitation gates, and sequential amplitude updates follow the
same order. This source must not be described as an unchanged arXiv v1 copy.
The figure PDFs and PNGs have been regenerated from the complete 56-task
statevector rerun performed in the current computing environment, using the
included [numerical tables](../results/paper/). Both figures use that same
dataset, including its 1.6 Å histories. They replace the arXiv v1 figure assets;
the original arXiv record is unchanged. The Results section describes the
rerun and iteration-capped points. Original figure styles are retained: the
PES has two panels with height ratio 1.8:1 and convergence has one panel.
Colors, line styles, markers, legend labels and order, axis labels and font
sizes follow the original PDF assets and plotting scripts.

[Figure provenance](figures/provenance.json) records the table hashes, upstream
dataset provenance, plotting-script hash, software versions and output hashes.
The figures use embedded PDF fonts, the original plot dimensions and the
original manuscript inclusion width. Regenerate them
from the repository root with the pinned Python environment:

```bash
python scripts/plot_paper_results.py results/paper --out-dir docs/figures
```

License and version comments have also been added to the TeX source.

When sharing or adapting these materials, credit the authors and preprint,
link to CC BY 4.0, and identify changes. The final preprint PDF is available
from arXiv rather than duplicated as a generated file in this repository.
Earlier drafts, a conference abstract draft and journal submission letters
are excluded from the public repository.

Original code, software documentation other than the manuscript, and numerical
tables are covered by the repository's [MIT License](../LICENSE). Third-party
software dependencies retain their respective licenses.

## Building the manuscript

With `latexmk`, REVTeX and the packages used by the source installed, run from
the repository root:

```bash
sh docs/build.sh main_rev3
```

Generated PDF and auxiliary files go to ignored `docs/build/main_rev3/`.
The scientific figure PDFs in `docs/figures/` are required source assets.
