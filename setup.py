import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', "r") as fp:
    install_requires = [pkg.replace("\n", "") for pkg in fp]

setuptools.setup(
    name="deepof",
    version="0.1.5",
    author="Lucas Miranda",
    author_email="lucas_miranda@psych.mpg.de",
    description="A suite for postprocessing time-series extracted from videos of freely moving rodents using DeepLabCut",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.mpcdf.mpg.de/lucasmir/deepof/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">3.8",
    platform="Platform independent",
    License="MIT",
    include_package_data=True,
    install_requires=install_requires,
)
