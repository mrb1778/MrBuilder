import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mrbuilder",
    version="0.0.1",
    author="Michael R. Bernstein",
    author_email="code@michaelrbernstein.com",
    description="Model and Repository Builder for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrb1778/mrbuilder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)