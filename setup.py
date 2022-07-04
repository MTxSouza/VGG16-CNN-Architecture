from setuptools import setup

setup(
    name="VGG16-Architecture",
    version="0.1.1",
    author="Matheus Oliveira de Souza",
    description="The VGG16 architecture, a neural net which classify images, made entirely with Tensorflow 2.x from scratch.",
    long_description=open("README.md", "r").read(),
    install_requires=[
        "tensorflow",
        "opencv-python",
        "matplotlib"
    ]
)