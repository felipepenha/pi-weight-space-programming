# Agent Rules & Guidelines

## Release & Version Alignment Rule

Before creating any git tag or publishing a release, the agent **MUST** ensure that all project configuration and source files are aligned to the target/future version number in a dedicated commit or PR.

Never create a release or git tag without first updating and verifying the version across all of the following locations:

1. **`package.json`**:
   - Update `"version": "X.Y.Z"`
2. **`pyproject.toml`**:
   - Update `version = "X.Y.Z"` under `[project]`
3. **`uv.lock`**:
   - Run `uv lock` at the repository root to regenerate and sync the lockfile to the updated `pyproject.toml` version.
4. **`extensions/header.ts`**:
   - Update the banner title and plain text string: `Pi Weight-Space Programming vX.Y.Z`
5. **`harness/wsp_engine.py`**:
   - Update the default download User-Agent string: `"User-Agent": "pi-weight-space-programming/X.Y.Z"`

### Verification Checklist Before Tagging
- Run `git grep "<old-version>"` to verify no stale version numbers remain in configuration or code.
- Ensure lockfiles (`uv.lock`, `*.lock`) and execution transcripts (`session.log`) are **not** ignored in `.gitignore`.
- Commit the version alignment changes before pushing the tag (`git tag <version>`) and publishing the release (`gh release create`).
