from setuptools import setup, find_packages

setup(
    name="your_package_name",              # Name of the package
    version="0.1.0",                      # Version number
    description="A brief description of your package",
    long_description=open('README.md').read(),  # Optional: Load long description from a file like README.md
    long_description_content_type="text/markdown",  # Ensure markdown formatting is interpreted
    author="Your Name",                   # Your name
    author_email="your.email@example.com", # Your email address
    url="https://github.com/yourusername/yourrepository", # Project URL
    packages=find_packages(),             # Automatically find and include all subpackages
    classifiers=[                         # Optional: Classifiers for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[                    # List your dependencies here
        "requests",                        # Example dependency
        "flask",                           # Example dependency
        "pandas",
        "numpy",
        "pyarrow"
    ],
    python_requires='>=3.6',               # Python version requirement
)
