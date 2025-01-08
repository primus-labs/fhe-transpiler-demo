import pickle

with open('./test/circut.bin', 'rb') as f:
    circuit = pickle.load(f)

with open('./test/cipher.bin', 'rb') as f:
    cipher = pickle.load(f)

eval_cipher = circuit.run(cipher)
eval_cipher.show()

with open('./test/eval_cipher.bin', 'wb') as f:
    pickle.dump(eval_cipher, f)