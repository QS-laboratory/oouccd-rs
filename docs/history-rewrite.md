# Public history and numerical provenance

Before publication, the history was rewritten to remove downloaded article
PDFs, submission correspondence, old manuscript drafts, generated manuscript
products, caches and a compiled development executable. Personal filesystem
paths and a private storage-bucket identifier were replaced in old text files.
Author identities and their recorded email addresses were retained.

The `main` branch retains the paper's sequential and L-BFGS implementation.
The `newton-shot-noise` branch retains subsequent research development.
The annotated `pre-newton-oo` tag still marks the same numerical baseline.

This changes commit identifiers, without changing the numerical Rust sources
at the corresponding historical commits. Historical result files continue to
record the original identifiers used when the calculations ran; they have not
been rewritten to suggest that a new calculation occurred. A mapping from
original to public commits accompanies this document after history cleanup.

A reproducible calculation also depends on recorded source patches, inputs,
lockfiles, environments and executable hashes. Rewriting history does not
rebuild the executable, rerun simulations or establish additional numerical
agreement beyond the checks in [verification.md](verification.md).
