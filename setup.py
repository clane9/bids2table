import pathlib

from setuptools import find_namespace_packages, find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


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
    description="ETL BIDS data into tables",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=packages,
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "hydra-core",
        "numpy",
        "pandas>=1.2.0",
        "pyarrow",
        "tabulate",
    ],
    extras_require={},
)
