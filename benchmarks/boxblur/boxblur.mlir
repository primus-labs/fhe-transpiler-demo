module {
  func.func @encryptedBoxBlur_8x8(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 8 {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst) -> (f32) {
          %1 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %arg5) -> (f32) {
            %2 = affine.load %arg0[((%arg2 + %arg6 - 1) * 8 + %arg3 + %arg4 - 1) mod 64] : memref<64xf32>
            %3 = arith.addf %arg7, %2 : f32
            affine.yield %3 : f32
          }
          affine.yield %1 : f32
        }
        affine.store %0, %arg1[(%arg2 * 8 + %arg3) mod 64] : memref<64xf32>
      }
    }
    return %arg1 : memref<64xf32>
  }
}
