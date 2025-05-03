from setuptools import setup, find_packages

setup(
    name='pytector',
    version='0.1.2',
    author='Max Melchior Lang',
    author_email='langmaxmelchior@gmail.com',
    description='A package for detecting prompt injections in text using Open-Source LLMs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MaxMLang/pytector',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'transformers>=4.0.0',
        'validators',
        'torch',
        'groq'
    ],
    extras_require={
        'gguf': ['llama-cpp-python>=0.2.0'],
        'test': ['pytest>=8.0'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
    python_requires='>=3.9',
)
