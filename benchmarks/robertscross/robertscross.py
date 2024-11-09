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
