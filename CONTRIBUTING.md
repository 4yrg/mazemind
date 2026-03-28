# Contributing to Mazemind

## Branch Naming

Use the format: `feature/<description>` or `fix/<description>`

Examples:
- `feature/maze-parser`
- `feature/dyna-q-agent`
- `fix/wall-collision-reward`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`

Examples:
- `feat(envs): implement MicromouseEnv with step/reward mechanics`
- `fix(agents): correct SARSA update using next_action`
- `test(parser): add wall configuration validation tests`

## Workflow

1. Create a feature branch from `main`
2. Implement your changes
3. Run tests: `pytest tests/`
4. Run linting if applicable
5. Commit with conventional commit messages
6. Push to remote
7. Merge to `main` with `--no-ff`

## Code Style

- Python 3.9+
- Type hints for function signatures
- Docstrings for public classes and methods
- No comments unless explicitly requested
- Follow existing patterns in the codebase
