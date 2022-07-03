from setuptools import setup

setup(
    name="VGG16-Architecture",
    version="0.1.0",
    author="Matheus Oliveira de Souza",
    description="The VGG16 architecture, a neural net which classify images, made entirely with Tensorflow 2.x from scratch.",
    long_description=open("README.md", "r").read(),
    install_packages=[
        "tensorflow",
        "opencv-python",
        "matplotlib"
    ]
)