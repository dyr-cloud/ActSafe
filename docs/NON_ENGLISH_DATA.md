# Language Audit

First-party Python, shell, Markdown, configuration, examples, and tests are scanned for CJK Unified Ideographs by `scripts/release_audit.py`. The final release scan must report zero matches.

Structured benchmark inputs are data rather than first-party prose and are not rewritten. In three frozen result files, literal CJK code points in inherited audit-reason strings are represented with standard JSON `\uXXXX` escapes. JSON decoding produces the original strings exactly, so all result values and metrics are unchanged while the release text scan reports no literal Chinese text in source, logs, scripts, configurations, or documentation.
