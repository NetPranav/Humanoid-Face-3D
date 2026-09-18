from setuptools import setup, find_packages

setup(
    name="face_geo_pipeline",
    version="0.1.0",
    description="Multi-view face geometry reconstruction pipeline producing neutral base meshes with micro-displacement and UE5-ready blendshapes",
    packages=find_packages(),
    python_requires=">=3.9",
)
