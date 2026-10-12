# Publishing releases

## Internal testing

Run **Trigger Release** on internal `main` to update `pyproject.toml`, commit the
version, and push its tag. Internal tag CI runs the test matrix, builds once,
and publishes packages to Artifactory. The tag is no longer automatically
copied to the public repository, so it can be tested internally first.

Commits on `main` still sync publicly after a successful build. This change
holds back release tags and public package publication, not source code.

## Promote a tag to public

1. Wait for internal tag CI to succeed and test the package from Artifactory.
2. In `Arm-Debug/vgf-adapter-model-explorer`, open
   **Actions > Sync release tag to public repo**.
3. Choose **Run workflow**, leave the workflow branch as `main`, and enter the
   existing internal tag, for example `v0.5.1`.

The workflow copies only that tag to `arm/vgf-adapter-model-explorer`, using
the existing `ARM_ORG_TOKEN`. It does not rebuild packages, create new tags
internally, or force-update existing public tags. An already identical public
tag is a no-op and does not trigger another public build.

The public tag push starts the existing public CI: test matrix, one build,
automatic PyPI publishing, then a GitHub release with packages and SBOMs.
Running the sync workflow is therefore the approval to start public release.
It does not check internal CI results for you; verify them before promotion.

The CLI equivalent is:

```bash
gh workflow run sync-release-tag.yml \
  --repo Arm-Debug/vgf-adapter-model-explorer \
  --ref main \
  -f release-tag=v0.5.1
```

Keep the existing PyPI Trusted Publisher configured for `ci.yml` and the
`pypi` environment. No new PyPI publisher configuration is required.

These instructions apply to tags created after the updated CI is merged.
Older tags retain their historical workflow definitions.
