from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='LaTRO',
    version='0.0.1',
    author="Haolin Chen",
    author_email="haolin.chen@salesforce.com",
    description="The official implementation of Latent Reasoning Optimization (LaTRO)",
    license="Apache License 2.0",
    url="https://github.com/SalesforceAIResearch/LaTRO",
    packages=find_packages(where="src"),
    install_requires=requirements,
)
