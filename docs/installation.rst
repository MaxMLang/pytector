Installation
============

Requirements
------------

* Python 3.9 or higher
* PyTorch (for Hugging Face models)
* Transformers library
* Validators library

Basic Installation
-----------------

Install pytector using pip:

.. code-block:: bash

   pip install pytector

This will install the core package with Hugging Face Transformers support.

Installation with GGUF Support
-----------------------------

To use GGUF models (like Llama models), install with the extra dependency:

.. code-block:: bash

   pip install pytector[gguf]

Or install the GGUF dependency separately:

.. code-block:: bash

   pip install llama-cpp-python>=0.2.0

Installation for Development
---------------------------

To install pytector for development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/MaxMLang/pytector.git
   cd pytector
   pip install -e .

For testing, install with test dependencies:

.. code-block:: bash

   pip install -e .[test]

Verifying Installation
---------------------

You can verify the installation by running:

.. code-block:: python

   import pytector
   print(pytector.__version__)

If you encounter any issues during installation, please check the :doc:`troubleshooting` section or open an issue on GitHub. 