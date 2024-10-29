module {
  func.func @encryptedRobertsCross_32x32(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) -> memref<1024xf32> {
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[((%arg2 - 1) * 32 + %arg3 - 1) mod 1024] : memref<1024xf32>
        %1 = affine.load %arg0[(%arg2 * 32 + %arg3) mod 1024] : memref<1024xf32>
        %2 = affine.load %arg0[((%arg2 - 1) * 32 + %arg3) mod 1024] : memref<1024xf32>
        %3 = affine.load %arg0[(%arg2 * 32 + %arg3 - 1) mod 1024] : memref<1024xf32>
        %4 = arith.subf %0, %1 : f32
        %5 = arith.mulf %4, %4 : f32
        %6 = arith.subf %2, %3 : f32
        %7 = arith.mulf %6, %6 : f32
        %8 = arith.addf %5, %7 : f32
        affine.store %8, %arg1[(%arg2 * 32 + %arg3) mod 1024] : memref<1024xf32>
      }
    }
    return %arg1 : memref<1024xf32>
  }
}
