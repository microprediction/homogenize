import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="homogenize",
    version="0.1.0",
    description="Survival probability for a regime-switching Ornstein-Uhlenbeck hazard: fast-switching expansion to any order",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/homogenize",
    author="microprediction",
    author_email="info@microprediction.org",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["homogenize"],
    python_requires=">=3.8",
    install_requires=["numpy"],
    extras_require={"test": ["pytest"]},
)
