from setuptools import setup, find_packages

setup(
    name="physics-ai-sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author="Yann LeCun",
    description="AI-driven physics simulation using Physics-Informed Neural Networks (PINNs).",
    url="https://github.com/YannLeCun25/physics-ai-sim",
)
