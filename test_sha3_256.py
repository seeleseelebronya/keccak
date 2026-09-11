""" Test the correctness of SHA3-256.
"""

from sha3_256 import sha3_256
import hashlib

MESSAGES = [b"", b"abc", b"hello", b"1234"]

def test_sa3_256():
    for message in MESSAGES:
        assert sha3_256(message) == hashlib.sha3_256(message).hexdigest()
