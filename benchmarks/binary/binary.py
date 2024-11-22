def binary8x8(arg0:list[float, 64], arg1:list[float, 64]):
    for i in range(4096):
        if arg0[i] > 125:
            arg1[i] = 255
        else:
            arg1[i] = 0
    return arg1