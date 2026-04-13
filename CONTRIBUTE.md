# How to contribute? 

We welcome bug reports, feature requests, and pull requests.
This project uses [uv](https://github.com/astral-sh/uv) for Python version, dependency, and project management.

## Development

We try to maintain good practices of readable open source code. Therefore, if you want to participate in the development and open your pool request, pay attention to the following points:

- To install the project with all development dependencies, run:
    ```shell
    uv sync --locked --all-extras --dev --group=scripts
    ```
  You can also use `pip` in your own Python environment, but using `uv` is the **preferred way** 
  cause of possible dependency resolve problems.  
    ```shell
    # dev version
    pip install -e ".[dev]"
    # scripts version for running examples
    pip install -e ".[scripts]"
    ```

- Before committing or pushing changes **run the formatters** from the repository root:
    ```shell
    uvx isort src tests && uvx black src tests
    ```

- To run tests locally with coverage enabled:
    ```shell
    uv run pytest --cov=. --cov-branch tests
    ```

- To add new dependencies, run:
    ```shell
    # prod version
    uv add <new-package>
    # dev version
    uv add --dev <new-package>
    # scripts version
    uv add --group=scripts <new-package>
    ```

- To buid and run docs locally, run:
    ```shell
    uv run pdoc --math -d google --no-include-undocumented -t .github/doc/ aiice
    ```

- To run any debug scripts with the project env, run:
    ```shell
    uv run <script.py> --group=scripts
    ```

- To run Jupyter notebooks with the project env, run:
    ```shell
    uv run --group=scripts --with jupyter jupyter lab    
    ```
 
## General tips
- Prefer contributing via forks, especially for external contributors
- Give only appropriate names to commits / issues / pull-requests

## Release process

Despite the fact that the framework is very small, we want to maintain its consistency. The release procedure looks like this:

- pull-request is approved by maintainers and merged with squashing commits
- a new tag is being released to the github repository and pypi with GitHub Actions
