module {
  func.func @binary64x64(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) -> memref<4096xf32> {
    affine.for %arg2 = 0 to 4096 {
      %0 = affine.load %arg0[%arg2] : memref<4096xf32>
      %cst = arith.constant 1.250000e+02 : f32
      %1 = arith.cmpf ogt, %0, %cst : f32
      scf.if %1 {
        %cst_0 = arith.constant 2.550000e+02 : f32
        affine.store %cst_0, %arg1[%arg2] : memref<4096xf32>
      } else {
        %cst_0 = arith.constant 0.000000e+00 : f32
        affine.store %cst_0, %arg1[%arg2] : memref<4096xf32>
      }
    }
    return %arg1 : memref<4096xf32>
  }
}
