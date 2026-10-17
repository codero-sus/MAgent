# Skill: debug a failing command or test

Diagnose the reported failure from evidence and repair the root cause.

1. Reproduce once with full, untruncated output and record the exact exit status.
2. Find the first application-owned frame or violated assertion, not merely the last exception line.
3. Read the real signatures and surrounding implementation before proposing a call or field.
4. Decide whether the test/assumption or implementation is wrong. State the contract used to decide.
5. Add the narrowest regression test, make a minimal fix, and run that test.
6. Run the nearest broader suite to detect collateral regressions.
7. Report changed paths, root cause, verification commands, and exact results.

Never hide a failing exit code with a pipe, retry a timed-out command unchanged, or claim success from inspection alone.
