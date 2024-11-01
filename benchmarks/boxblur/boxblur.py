def encryptedBoxBlur_8x8(arg0:list[float,64], arg1:list[float, 64]):
    for x in range(8):
        for y in range(8):
            value = 0.0
            for j in range(3):
                for i in range(3):
                    value += arg0[((x+i-1)*8+y+j-1)%64 ]
            arg1[(8*x+y)%64] = value
    return arg1
