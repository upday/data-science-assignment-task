import os

from setuptools import setup, find_packages
from setuptools.command.develop import develop as Develop
from setuptools.command.install import install as Install


def models_installation():
    import spacy.cli.download as download

    download("en_core_web_md")  # medium


class CustomInstall(Install):
    def run(self):
        Install.run(self)
        models_installation()


class CustomDevelop(Develop):
    def run(self):
        Develop.run(self)
        models_installation()


# Read requirements
requirements_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
)
with open(requirements_file, "r") as f:
    requirements = f.read().splitlines()

# Package configuration
setup(
    name="textclf",
    version="0.0.1",
    description="Text Classification Task",
    include_package_data=True,
    setup_requires=["wheel"],
    packages=find_packages(),
    install_requires=requirements,
    cmdclass={"install": CustomInstall, "develop": CustomDevelop},
)
