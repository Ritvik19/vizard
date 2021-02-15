import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vizard",
    version="0.1.1",
    author="Ritvik Rastogi",
    author_email="rastogiritvik99@gmail.com",
    description="Intuitive, Easy and Quick Vizualizations for Data Science Projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ritvik19",
    packages=setuptools.find_packages(
        exclude=[".git", ".idea", ".gitattributes", ".gitignore", ".github"]
    ),
    install_requires=[
        "numpy>=1.18.1",
        "pandas>=1.0.3",
        "matplotlib>=3.1.3",
        "scipy>=1.4.1",
        "seaborn>=0.11.1",
        "wordcloud==1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)