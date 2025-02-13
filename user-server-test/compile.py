import fhecomplr
import pickle

def encryptedBoxBlur_64x64(arg0:list[float,4096], arg1:list[float, 4096]):
    for x in range(64):
        for y in range(64):
            value = 0.0
            for j in range(3):
                for i in range(3):
                    value += arg0[((x+i-1)*64+y+j-1)%4096 ]
            arg1[(64*x+y)%4096] = value
    return arg1

def encryptedRobertsCross_64x64(img:list[float, 4096], output:list[float, 4096]):
    for x in range(64):
        for y in range(64):
            val1 = img[((x - 1) * 64 + (y - 1)) % 4096]
            val2 = img[(x * 64 + y) % 4096]
            val3 = img[((x - 1) * 64 + y) % 4096]
            val4 = img[(x * 64 + (y - 1)) % 4096]
            diff1 = (val1 - val2)*(val1 - val2)
            diff2 = (val3 - val4)*(val3 - val4)
            output[(x * 64 + y) % 4096] = diff1 + diff2
    return output

def binary64x64(arg0:list[float, 4096], arg1:list[float, 4096]):
    for i in range(4096):
        if arg0[i] > 125:
            arg1[i] = 255
        else:
            arg1[i] = 0
    return arg1


compiler = fhecomplr.Compiler()
circuit, rotate_steps = compiler.compile(encryptedRobertsCross_64x64) # OpenFHE needs rotate steps to generate the evaluate keys.

with open('./test/circut.bin', 'wb') as f:
    pickle.dump(circuit, f)
with open('./test/rotate_steps.bin', 'wb') as f:
    pickle.dump(rotate_steps, f)