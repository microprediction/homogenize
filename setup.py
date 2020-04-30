import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="homogenize",
    version="0.0.1",
    description="Survivial probability estimator for Regime Switching Orstein-Uhlenbeck triply stochastic process",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/homogenize",
    author="microprediction",
    author_email="info@microprediction.org",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['homogenize'],
    test_suite='pytest',
    tests_require=['pytest','numpy','requests','pathlib'],
    include_package_data=True,
    install_requires=["numpy",'numpy','requests','pathlib'],
    entry_points={
        "console_scripts": [
            "homogenize=homogenize.__main__:main",
        ]
     },
     )
