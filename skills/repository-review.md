# Skill: repository review

Review the current workspace for defects that could cause incorrect behavior, security exposure, data loss, deadlocks, protocol violations, or misleading operator output.

1. Read the repository structure and its contribution/test instructions.
2. Inspect changed files first, then their callers and tests.
3. Verify every finding against the actual implementation; do not infer an API from its name.
4. Rank findings by severity. For each finding give the exact path and line, the failure scenario, and a minimal repair.
5. Run the narrowest existing verification that can confirm or reject each suspected defect.
6. Do not modify files unless the user explicitly asked for fixes.

Done means either: concrete, reproducible findings are listed worst-first; or the response says no finding was confirmed and names the checks performed. Do not add generic praise or speculative concerns.
