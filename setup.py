import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mrbuilder",
    version="1.0.0",
    author="Michael R. Bernstein",
    author_email="code@michaelrbernstein.com",
    description="A builder for maintainable, extensible, and cross platform deep learning model creation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrb1778/mrbuilder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research'
    ],
    keywords='deep learning, neural networks, machine learning, '
             'scientific computations, tensorflow, keras, pytorch',
    install_requires=[
        # no run-time or installation-time dependencies
    ],
)