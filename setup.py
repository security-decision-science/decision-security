from setuptools import setup

# Map the package name to the 'src' directory, which already has __init__.py and your modules.
setup(
    packages=["decision_security"],          # the install-time package name
    package_dir={"decision_security": "src"} # where its code lives on disk
)