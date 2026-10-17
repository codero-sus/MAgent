# Skill: release readiness check

Determine whether the current MAgent checkout is ready to release.

1. Read version/package metadata, README, license, build configuration, CI, and repository status.
2. Run the stdlib test suite, offline smoke, safe-tool smoke, and improvement verifier.
3. Build wheel and sdist in an isolated environment and inspect their contents.
4. Confirm the wheel includes `py.typed`, the web console, and packaged documentation.
5. Import MAgent from the built wheel outside the repository and run `magent version` plus one offline task.
6. Check that user-facing prose spells MAgent exactly and that credentials/build artifacts are not tracked.
7. Report blockers first, with exact command and output. If clear, list all evidence used.

Do not publish, tag, push, or upload anything unless separately and explicitly requested.
