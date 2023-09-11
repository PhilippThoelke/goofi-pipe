"""
Connection serves as a type for checking if a given object is a multiprocessing Connection.
Linux implements multiprocessing.connection.Connection, Windows implements
multiprocessing.connection.PipeConnection, but both inherit from _BaseConnection. While not
ideal, this workaround allows us to use isinstance(obj, goofi.connection.Connection) on both
platforms.
"""
from multiprocessing.connection import _ConnectionBase as Connection  # noqa: F401
