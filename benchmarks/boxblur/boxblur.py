def encryptedBoxBlur_64x64(arg0:list[float,4096], arg1:list[float, 4096]):
    for x in range(64):
        for y in range(64):
            value = 0.0
            for j in range(3):
                for i in range(3):
                    value += arg0[((x+i-1)*64+y+j-1)%4096 ]
            arg1[(64*x+y)%4096] = value
    return arg1
