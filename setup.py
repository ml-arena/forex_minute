from setuptools import setup, find_packages

setup(
    name="forex_minute",
    version="0.1",
    description="Forex Minute Environment",
    packages=find_packages(),
    package_data={
        'forex_minute.env': ['*.png'],
    },
    install_requires=[],
    python_requires=">=3.9",
)