import pickle

with open('./test/cryptor.bin', 'rb') as f:
    cryptor = pickle.load(f)

with open('./test/eval_cipher.bin', 'rb') as f:
    eval_cipher = pickle.load(f)

output_image = cryptor.decrypt(eval_cipher)
output_image.show()
output_image.save("./test/evaled.png")