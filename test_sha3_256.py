from sha3_256 import sha3_256
import hashlib

input = b""

print("my lib:     ", sha3_256(input))
print("offical lib:", hashlib.sha3_256(input).hexdigest())
