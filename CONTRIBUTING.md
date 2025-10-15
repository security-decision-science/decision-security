# Contributing to security-decision-science

👋 Thanks for your interest! Contributions are welcome—bug reports, docs, tests, and features.

## Ways to Contribute
- File issues for bugs or improvements
- Submit pull requests (PRs)
- Improve documentation

## Development Setup
- Python 3.11+ recommended
- Create a virtualenv:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements-dev.txt -r requirements.txt
  ```
- Run tests (if present):
  ```bash
  pytest -q
  ```

## Branching & Commit Messages
- Branch: feat/*, fix/*, docs/*, chore/*, ci/*
- Commits: use conventional style, e.g. fix(auth): handle empty token
- Keep commits focused and small

## Pull Requests
- Base branch: master
- Ensure CI is green
- Explain what and why in the PR description
- Add tests when fixing bugs or adding features
- Security impact: call out if changes could affect auth, secrets, or supply chain

## Code Review
- At least one review approval is required
- Address review comments via follow-up commits (avoid force-push on public PRs)

## Security
- Do not file public issues for vulnerabilities
- Follow our SECURITY.md to report privately

## Code of Conduct

By participating, you agree to follow our Code of Conduct.

## License

By contributing, you agree your contributions are licensed under the repository’s MIT License.
