from setuptools import setup, find_packages

# Delegate metadata to pyproject.toml; just control package discovery here.
setup(
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["decision_security", "decision_security.*"]),
)