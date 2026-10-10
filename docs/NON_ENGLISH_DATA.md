# Language Audit

First-party Python, shell, Markdown, configuration, examples, and tests are scanned for CJK Unified Ideographs by `scripts/release_audit.py`. The final release scan must report zero matches.

Structured benchmark inputs are data rather than first-party prose and are not rewritten. Precomputed result files are not part of the repository. The release text scan therefore covers source, scripts, configurations, documentation, and retained benchmark inputs without depending on generated outputs.
