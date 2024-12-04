# Heavily Damaged Image Reconstructor (HDIR)

This external component of the **Meningioma** project was originally developed by **Ezequiel López Rubio**. I do not claim authorship of the original code. My modifications are limited to adapting and integrating the code into the pipeline of this project to meet specific requirements.

**Changelog:**

- Compiled the C libraries to run in 64-bit Linux SO (Ubuntu 22.04 LTS):

```bash
/usr/local/MATLAB/R2024b/bin/mex -v -L/usr/lib/x86_64-linux-gnu -lblas -llapack -lf2c -I. ClassicKernelRegression.c MatesLap.c Debugging.c
/usr/local/MATLAB/R2024b/bin/mex -v -L/usr/lib/x86_64-linux-gnu -lblas -llapack -lf2c -I. SteeringKernelRegression.c MatesLap.c Debugging.c
/usr/local/MATLAB/R2024b/bin/mex -v -L/usr/lib/x86_64-linux-gnu -lblas -llapack -lf2c -I. SteeringMatrix.c MatesLap.c Debugging.c
```

- Updated `SteeringMatrix.c` to ensure compatibility with 64-bit systems. These changes address potential integer size mismatches on 64-bit systems, ensuring proper memory allocation and array indexing:
  - Replaced `int` with `mwSize` for variables related to array dimensions (`N`, `M`, `radius`, etc.) and loop indices.
  - Ensured `mwSize` is used for all variables passed to MEX API functions like `mxMalloc`, `mxCreateNumericArray`, and `memcpy`.
  - Adjusted input retrieval (e.g., `wsize`) to use `mwSize` for consistency.

- The `SteeringMatrix.c` script requires the Matlab _Image Processing Toolbox_.

## References

LÓPEZ-RUBIO, Ezequiel. Restoration of images corrupted by Gaussian and uniform impulsive noise. *Pattern Recognition*, 2010, vol. 43, no 5, p. 1835-1846.

