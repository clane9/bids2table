import pathlib

from setuptools import find_namespace_packages, find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
    "hydra-core",
    "numpy",
    "pandas",
    "pyarrow",
    "tabulate",
    "pyyaml",
]

# NOTE: these dependencies also pinned in .pre-commit-config.yaml
DEV_REQUIRES = [
    "black==22.10.0",
    "flake8==5.0.4",
    "isort==5.10.1",
    "mypy==0.982",
    "pre-commit",
    "pylint>=2.5.0",
    "pytest",
    "types-tabulate",
    "types-PyYAML",
]

EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRES,
}


def get_version():
    path = HERE / "bids2table" / "__init__.py"
    py = open(path, "r").readlines()
    version_line = [l.strip() for l in py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


packages = find_packages(include=["bids2table*", "tests*"]) + find_namespace_packages(
    include=["hydra_plugins.*"]
)

setup(
    name="bids2table",
    version=get_version(),
    author="Connor Lane",
    license="MIT",
    url="https://github.com/clane9/bids2table",
    description="Organize neuroimaging data derivatives into parquet tables",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=packages,
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": ["bids2table = bids2table:__main__._main"],
    },
)
