from setuptools import setup, find_packages

setup(
    name="mini2_rl",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "matplotlib>=3.5"],
    extras_require={"drl": ["torch>=2.0"]},
    author="Ismail",
    description="EECS 590 Mini-2 RL repository (windy chasm / lidar / circuit).",
)
