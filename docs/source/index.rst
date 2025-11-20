.. gais documentation master file, created by
   sphinx-quickstart on Thu Nov 20 00:11:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`gais`: Generative AI Suite 
===========================

A unified and consistent interface for working with multiple LLM providers.

**`gais` provides:** 

* **Parallel multi-prompt generation:** run several prompts at once using a single provider.
* **Multi-provider generation:** send one prompt to multiple providers and collect all responses together.
* **Structured outputs with Pydantic:** clean, validated responses using Pydantic models.
* **Universal parsing support:** uses native parsing when available (e.g., OpenAI .parse), and an automatic fallback parser when it isn’t.

Currently supported providers: **OpenAI, DeepSeek, Groq, Anthropic, Fireworks, Gemini, and Ollama.**

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api_reference


Cite this Project
-----------------

If you use ``gais`` in your work, please cite it as:

::

  @misc{gais,
    title        = {gais},
    author       = {Hanif Sajid},
    year         = {2025},
    month        = November,
    version      = {0.1.0},
    howpublished = {https://github.com/hanifsajid/gais},
    note         = {MIT License}
     }
