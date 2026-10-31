"""shilp — a small, readable model server.

The package is layered, and each layer only talks to the one below it:

    cli  ->  hub / modify  ->  store  ->  engine  ->  server

`store` is the single source of truth on disk. `engine` turns a stored model
into tokens. `server` exposes those tokens over two HTTP surfaces that share
one code path.
"""

__version__ = "0.1.0"
