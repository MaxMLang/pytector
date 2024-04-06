from setuptools import setup, find_packages

setup(
    name='pytector',
    version='0.0.1',
    author='Max Melchior Lang',
    author_email='langmaxmelchior@gmail.com',
    description='A package for detecting prompt injections in text using Open-Source LLMs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MaxMLang/pytector',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.0.0',
        'validators'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    test_suite='tests',
)
