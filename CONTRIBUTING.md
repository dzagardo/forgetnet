# Contributing to ForgetNet 🤝

First off, thank you for considering contributing to ForgetNet! It's people like you that make ForgetNet such a great tool. 🌟

## Code of Conduct 📜

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute? 🤔

### Reporting Bugs 🐛

- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/dzagardo/forgetnet/issues).
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/dzagardo/forgetnet/issues/new). Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements 💡

- Open a new issue with a clear title and detailed description of the suggested enhancement.
- Provide any relevant examples or mock-ups that could help explain your suggestion.

### Pull Requests 🚀

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup 🛠️

To set up ForgetNet for local development:

1. Fork the [ForgetNet](https://github.com/dzagardo/forgetnet) repo on GitHub.
2. Clone your fork locally:
    ```
    git clone git@github.com:dzagardo/forgetnet.git
    ```
3. Create a branch for local development:
    ```
    git checkout -b name-of-your-bugfix-or-feature
    ```
4. Install the development dependencies:
    ```
    pip install -e ".[dev]"
    ```
5. Make your changes locally.
6. Run the tests to ensure your changes haven't broken anything:
    ```
    pytest
    ```
7. Commit your changes and push your branch to GitHub:
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```
8. Submit a pull request through the GitHub website.

## Coding Style 🎨

- We use [black](https://github.com/psf/black) for code formatting. Please ensure your code is formatted with black before submitting a pull request.
- We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.
- We use [type hints](https://docs.python.org/3/library/typing.html) wherever possible.

## Testing 🧪

We use [pytest](https://docs.pytest.org/en/stable/) for testing. Please write tests for any new code you create.

## Documentation 📚

We use [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. Please ensure any new functions or classes you add are properly documented.

## Questions? 🤔

Don't hesitate to reach out if you have any questions about contributing. You can open an issue or contact the maintainers directly.

Thank you for your interest in improving ForgetNet! 🙏