#!/usr/bin/env python3
"""The bare minimum: run a language model with no llama.cpp, and serve it.

This is the reference implementation for docs/00-theory.md. It is deliberately
naive — PyTorch, a tokenizer, a sampling loop, and FastAPI — because its job is
