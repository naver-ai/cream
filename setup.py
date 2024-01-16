"""Cream
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license"""
import os

from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "cream", "_version.py")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    return data["__version__"]


def read_long_description():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


setup(
    name="cream-python",
    version=read_version(),
    license="MIT",
    description="Visually-Situated Natural Language Understanding with Contrastive Reading Model and Frozen Large Language Models",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Geewook Kim, Hodong Lee, Daehee Kim, Haeji Jung, Sanghee Park, Yoonsik Kim, Sangdoo Yun, Taeho Kil, Bado Lee, Seunghyun Park",
    author_email="gwkim.rsrch@gmail.com",
    packages=find_packages(
        exclude=[
            "tutorial",
            "lightning_module.py",
            "train.py",
            "test.py",
            "pyproject.toml",
        ]
    ),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers==4.34.1",
        "timm>=0.8.3.dev0",
        "pytorch-lightning>=1.8.6,<=1.9.5",
        "sentencepiece",
        "Pillow",
        "lmdb",
    ],
)
