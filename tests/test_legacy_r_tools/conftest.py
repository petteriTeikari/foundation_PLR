"""
Local conftest for legacy R tools tests.
This file prevents pytest from loading the parent conftest.py which has
heavy dependencies that may not be installed.
"""
