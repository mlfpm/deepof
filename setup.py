import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    install_requires = [pkg.replace("\n", "") for pkg in fp]

# Add system specific dependencies
install_requires += [
    "tensorflow>=2.2.0;platform_machine != 'arm64' or platform_system != 'Darwin'",
    "tensorflow-macos>=2.5.0;platform_machine == 'arm64' and platform_system == 'Darwin'",
    "keras-tcn==3.5.0;platform_machine != 'arm64' or platform_system != 'Darwin'",
    "keras-tcn-macos==1.0;platform_machine == 'arm64' and platform_system == 'Darwin'",
]

setuptools.setup(
    name="deepof",
    version="0.2",
    author="Lucas Miranda",
    author_email="lucas_miranda@psych.mpg.de",
    description="A suite for postprocessing time-series extracted from videos of freely moving rodents using DeepLabCut",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.mpcdf.mpg.de/lucasmir/deepof/",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
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
