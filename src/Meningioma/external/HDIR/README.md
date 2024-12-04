# Heavily Damaged Image Reconstructor (HDIR)

This external component of the **Meningioma** project was originally developed by **Ezequiel López Rubio**. I do not claim authorship of the original code. My modifications are limited to adapting and integrating the code into the pipeline of this project to meet specific requirements.

**Changelog:**

- Compiled the C libraries to run in 64-bit Linux SO (Ubuntu 22.04 LTS):

```bash
/usr/local/MATLAB/R2024b/bin/mex -v -L/usr/lib/x86_64-linux-gnu -lblas -llapack -lf2c -I. ClassicKernelRegression.c MatesLap.c Debugging.c
/usr/local/MATLAB/R2024b/bin/mex -v -L/usr/lib/x86_64-linux-gnu -lblas -llapack -lf2c -I. SteeringKernelRegression.c MatesLap.c Debugging.c
/usr/local/MATLAB/R2024b/bin/mex -v -L/usr/lib/x86_64-linux-gnu -lblas -llapack -lf2c -I. SteeringMatrix.c MatesLap.c Debugging.c
```


## References

LÓPEZ-RUBIO, Ezequiel. Restoration of images corrupted by Gaussian and uniform impulsive noise. *Pattern Recognition*, 2010, vol. 43, no 5, p. 1835-1846.

