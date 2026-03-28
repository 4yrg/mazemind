from setuptools import setup, find_packages

setup(
    name="mazemind",
    version="0.1.0",
    description="Tabular RL for Micromouse maze pathfinding: Dyna-Q vs SARSA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "ui": ["streamlit>=1.28.0"],
        "dev": ["pytest>=7.4.0"],
    },
)
