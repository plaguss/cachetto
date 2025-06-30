# cachetto
Disk-based caching for functions returning pickleable objects and pandas DataFrames, plain and simple.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cachetto)](https://pypi.org/project/cachetto)
[![PyPI](https://img.shields.io/pypi/v/cachetto)](https://pypi.org/project/cachetto)
[![Tests](https://github.com/plaguss/cachetto/actions/workflows/ci.yaml/badge.svg)](https://github.com/plaguss/cachetto/actions/workflows/ci.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


> [!WARNING]
>
> cachetto is experimental, the API is subject to changes.

## Getting Started

This is a simple library, but it can be handy for those who had to deal with codebases that have functions that take it's time to generate or process tabular data in the form of dataframes, either due to slow computations or queries. If that may be your case, take a look at the usage to see if you may find some help here.

Features:

- Seamless caching for functions or methods returning that can be pickled, including pandas dataframes

- Customizable cache directory

- Cache expiration with invalid_after (e.g., "1d", "6h")

- Toggle caching on or off

- Uses pickle to serialize the data

### Installation

`cachetto` is available on PyPI, and can be installed with:

```shell
# Using uv
uv add cachetto
# Using pip
pip install cachetto
```

The only required dependency is `pandas>=1.5.3` and Python 3.10 or higher.

### Usage

The API consists basically of a single decorator `cached`.

#### Minimal usage (No config)

Just decorate your function. By default, it uses an internal cache directory and never invalidates:

```py
from cachetto import cached
import pandas as pd

@cached
def get_data():
    print("Running expensive computation...")
    return {"df": pd.DataFrame({"value": range(10)}), "meta": ("some data", 1)}

result = get_data()  # Will run and cache
result = get_data()  # Will load from cache
```

#### Custom cache directry

Specify where cached files should be stored:

```py
@cached(cache_dir="cache_files")
def load_big_dataframe():
    return pd.DataFrame({"big": range(100000)})
```

#### Add cache expiration

Expire the cache after a certain duration (e.g., 1 day, 3 hours):

```py
@cached(cache_dir="cache_files", invalid_after="1d")
def get_fresh_data():
    return pd.DataFrame({"timestamp": [pd.Timestamp.now()]})
```

*If the cached file is older than 1 day, the function will re-run and overwrite the cache.*

#### Temporarily disable caching

Use the `caching_enabled` flag to bypass cache logic (e.g., for debugging, when running on a different environment):

```py
@cached(caching_enabled=False)
def debug_function():
    print("No caching here")
    return pd.DataFrame({"x": range(3)})
```

#### Clear cached files manually

You can programmatically clear the cache for a decorated function:

```py
@cached
def some_data():
    return pd.DataFrame({"numbers": [1, 2, 3]})

some_data.clear_cache()  # Deletes all cached files for this function
```

#### Use with class methods

Works equally with class methods:

```py
class MyModel:
    @cached(cache_dir="model_cache")
    def load_data(self):
        return pd.DataFrame({"model": ["A", "B", "C"]})
```

#### Get the name of the file loaded

In case you want to keep track of files loaded from cache, use the verbose option:

```py
@cached(verbose=True)
def get_data_verbose():
    return pd.DataFrame({"timestamp": [pd.Timestamp.now()]})

get_data_verbose()
get_data_verbose()
```

## Development

### Tests

Every new feature must include the corresponding tests, and ensure the coverage statys at 100% for the CI job to succeed:

```shell
make unit-tests       # Tests while developing with the default version
make cov-tests        # Check the coverage (html report generated)
make test-all-python  # Runs the tests with all the supported python versions
```

### Lint

Pre-commit is integrated for linting and formatting, and additionally, `mypy` must be run to ensure the typing is correct:

```shell
make typecheck-mypy
```

### Release

Locally, running `make release` will build and release the package in PyPI, but the CI is prepared to do it. Bump the version accordingly, create a new tag, and push it, this will trigger the release job:

```shell
uv version --bump [BUMP]
git tag v[NEW VERSION]
git push origin v[NEW VERSION]
```

## License

This repository is licensed under the [MIT License](https://github.com/plaguss/cachetto/blob/main/LICENSE).

## Credits

It's heavily inspired by [cachier](https://github.com/python-cachier/cachier), but with a builtin support for pandas dataframes, and just disk-based caching based on pickle.
