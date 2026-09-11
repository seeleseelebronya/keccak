from sha3_256 import sha3_256
import hashlib

MESSAGES = [b"", b"abc", b"hello", b"1234"]

for message in MESSAGES:
    assert sha3_256(message) == hashlib.sha3_256(message).hexdigest()
