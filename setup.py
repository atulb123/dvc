from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
setup(
    name="src",
    version="0.0.1",
    author="atulb123",
    description="A small package for dvc ml pipeline demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atulb123/dvc",
    author_email="atulmysuru@gmail.com",
    # package_dir={"": "src"},
    # packages=find_packages(where="src"),
    packages=["src"],
    license="GNU",
    python_requires=">=3.6",
    install_requires=[
        'dvc',
        'dvc[gdrive]',
        'dvc[s3]',
        'pandas',
        'scikit-learn'
    ]
)
