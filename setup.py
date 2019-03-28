from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="merf",
    version="0.3",
    description="Mixed Effects Random Forest",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="random forest machine learning mixed effects",
    url="https://github.com/manifoldai/merf",
    author="Manifold, Inc.",
    author_email="sdey@manifold.ai",
    license="MIT",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "scikit-learn"],
    include_package_data=True,
    zip_safe=False,
)
