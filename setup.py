from setuptools import setup, find_packages
setup(
    name="rapidfire_project_meta",
    version="0.0.1",
    packages=find_packages(exclude=["tests*"]),
    description="RapidFire project scaffolding (scripts, serving, SFT).",
    author="Shubham",
)
