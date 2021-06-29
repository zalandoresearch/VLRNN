from setuptools import setup

setup(
    name='vlrnn',
    version='0.0.1',    
    description='Very Long Recurrent Neural Networks',
    url='https://github.com/zalandoresearch/VLRNN',
    author='Roland Vollgraf',
    author_email='roland.vollgraf@zalando.de',
    license='MIT',
    packages=['vlrnn'],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.8.0"
    ],
    test_suite="tests",
)
