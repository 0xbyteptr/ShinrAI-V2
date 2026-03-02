#!/usr/bin/env python3
"""Legacy entrypoint for the Shinrai chatbot.

This script now simply delegates to the package command-line
interface located in :mod:`shinrai.cli`.  The main logic has been
split across several modules inside the ``shinrai`` package in order
to improve maintainability and readability.

Usage remains:

    python shinrai.py train --url ...
    python shinrai.py chat

but you may also invoke the package directly with
``python -m shinrai``.
"""

from shinrai.cli import main

if __name__ == "__main__":
    main()
