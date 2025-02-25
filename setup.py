import os
import os.path as osp
from setuptools import setup, find_packages

with open('README.md') as fin:
    lines = []
    for line in fin:
        if line.startswith("<"):
            continue 
        
        if "## Examples" in line:
            break
        
        lines.append(line)
        
    long_description = "".join(lines)


with open("sfeno/VERSION", "rt") as fin:
    version = fin.read().strip()

setup (
    name="sfeno",
    version=version,
    description="SFENO: Signal Flow Estimation based on Neural ODE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cxinsys/sfeno",
    author="Jung Hoi Hur, Daewon Lee",
    author_email=" kmo73724@gmail.com, daewon4you@gmail.com",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(),
    package_data={
        "": ["VERSION", "*.csv", "*.json"],
        # "sfeno.resources": ["icon.png", "logo.png"],
    },
    # scripts=scripts,
)
