import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# hacky, but...
# https://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py
with open("requirements.txt", "r") as f:
    install_reqs = [line for line in f.readlines() if not line.startswith("#")]

setuptools.setup(
    name="hfseqio",
    version="0.0.1",
    description="HFSeqIO: Interop for HF datasets and SeqIO",
    url="https://github.com/dlwh/hfseqio",
    author="David Hall",
    author_email="dlwh@cs.stanford.edu",
    license="BSD 3-Clause",
    install_requires=install_reqs,
    long_description=long_description,
    packages=setuptools.find_packages(where="src", exclude=("tests",)),
    # https://stackoverflow.com/questions/70777486/pip-install-e-doesnt-allow-to-import-via-package-dir
    package_dir={"": "src/"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS X",
        "Programming Language :: Python :: 3.10",
    ],
)
