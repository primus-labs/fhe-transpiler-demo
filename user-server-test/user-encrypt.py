import fhecomplr
import pickle

with open('./test/rotate_steps.bin', 'rb') as f:
    rotate_steps = pickle.load(f)

cryptor = fhecomplr.Cryptor(rotate_steps)
img = cryptor.read("./benchmarks/binary/test.png")
img.show()

cipher = cryptor.encrypt(img)
cipher.show()

with open('./test/cipher.bin', 'wb') as f:
    pickle.dump(cipher, f)
with open('./test/cryptor.bin', 'wb') as f:
    pickle.dump(cryptor, f)