# Fake_News_Detection

## Overview

A binary text classifier that labels news articles as **FAKE** or **REAL** using TF-IDF feature extraction and a Passive Aggressive linear classifier. The entire pipeline — data loading, training, artifact persistence, and evaluation — runs inside a single Jupyter notebook.

## Tech Stack

- Python (requirements.txt based)

## Repository Structure

- `.gitattributes`
- `.gitignore`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `Fake_news_Detection.ipynb`
- `LICENSE`
- `news/`
- `README.md`
- `requirements.txt`
- `SECURITY.md`
- `TEST_REPORT.md`
- ... and 1 more entries

## Getting Started

### Prerequisites

- Git
- Runtime dependencies for this project's stack

### Installation

```bash
uv venv
uv pip install -r requirements.txt
```

## Usage

Use the project's documented entrypoint (CLI/app script) from this repository.

## Testing

Run tests with `uv run pytest` from repository root.

## Security

Please review [SECURITY.md](SECURITY.md) for reporting and handling security issues.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening issues or pull requests.

## Changelog

Ongoing changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
