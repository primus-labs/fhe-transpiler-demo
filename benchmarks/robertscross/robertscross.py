def encryptedRobertsCross_32x32(img:list[float, 1024], output:list[float, 1024]):
    for x in range(32):
        for y in range(32):
            val1 = img[((x - 1) * 32 + (y - 1)) % 1024]
            val2 = img[(x * 32 + y) % 1024]
            val3 = img[((x - 1) * 32 + y) % 1024]
            val4 = img[(x * 32 + (y - 1)) % 1024]
            diff1 = (val1 - val2)*(val1 - val2)
            diff2 = (val3 - val4)*(val3 - val4)
            output[(x * 32 + y) % 1024] = diff1 + diff2
    return output
