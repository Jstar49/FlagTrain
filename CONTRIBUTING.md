# Contributing to FlagTrain

FlagTrain follows the
[FlagOS contribution guidelines](https://github.com/flagos-ai/community/blob/main/contributors/CONTRIBUTING.md)
and [Code of Conduct](https://github.com/flagos-ai/community/blob/main/CODE_OF_CONDUCT.md).
This repository is in its initial setup stage. Contributions to Triton training
operators, component support, tests, benchmarks, and documentation are welcome.

## Propose a change

Search existing issues and pull requests before opening a new one. Describe the
environment, expected behavior, and reproduction steps for a bug. Discuss substantial changes before
implementation; cross-project proposals should follow the
[FlagOS FEP process](https://github.com/flagos-ai/community/tree/main/fep).

Report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Work on a branch

Fork and clone the repository, or use your existing checkout if you have write
access. Create a separate branch for each change. Keep changes focused and update
relevant documentation alongside user-facing behavior.

## Implement and validate

Place implementations under `src/flag_train/megatron/`,
`src/flag_train/deepspeed/`, or `src/flag_train/transformer_engine/`.
Keep corresponding validation and examples in the same component directories
under `tests/`, `benchmark/`, and `examples/`.

Include a reference implementation, forward and gradient checks as applicable,
numerical tolerances, and reproducible performance measurements. Document the
component versions and hardware used. See [development guidance](docs/development.md).

Dependency changes follow the [installation guide](docs/installation.md):
the shared stack (torch, flagtree, numpy, vendor runtimes) belongs in
[build-infra](https://github.com/flagos-ai/build-infra), repo-wide Python
packages go in `pyproject.toml`, and vendor-specific pins go in `backends.yaml`
followed by `python3 tools/sync_pyproject_extras.py --write`. Do not add a
repository Dockerfile for dependencies the public images can provide.

Add executable test instructions alongside each implementation. Describe the
validation performed in your pull request, including any checks that could not
be run and why.

## Commit and sign off

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
and PR titles, for example `docs: clarify contribution workflow` or
`fix: correct configuration validation`.

FlagOS requires a [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) sign-off for code contributions. Use your own configured name and email and
sign off only contributions you have the right to submit:

```sh
git commit -s -m "docs: clarify contribution workflow"
```

The `-s` option adds a `Signed-off-by` line to the commit message. Preserve the
sign-off when amending or rebasing commits.

## Open a pull request

Target `main`. Explain the problem, the change,
validation, related issues, and any compatibility impact. Review the diff to
ensure it contains only intended changes and no credentials or private data.

Under the FlagOS review policy, required checks must pass and at least one
appropriate SIG Approver must approve before merge. Follow the
[FlagOS code review guide](https://github.com/flagos-ai/community/blob/main/contributors/review-guide.md)
and respond to review feedback. If reviewers have not yet been assigned during
the bootstrap stage, use the routing guidance in the organization contribution
guidelines.

PRs use squash merge. The PR title becomes the squash commit message and must
follow Conventional Commits. The approver handling the merge should preserve
the required DCO sign-off.
