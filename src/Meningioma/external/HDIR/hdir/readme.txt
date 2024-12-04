Heavily Damaged Image Reconstructor (HDIR)
------------------------------------------

This is experimental software. It is provided for noncommercial research purposes only. Use at your own risk. No warranty is implied by this distribution.

Reference:
L�pez-Rubio, E. (2010). Restoration of images corrupted by Gaussian and uniform impulsive noise. Pattern Recognition 43(5), pp. 1835�1846.

There are five kinds of files:
-Files with .m extension: they are Matlab files which form the top level of the implementation.
-Files with .c and .h extension: they are the C MEX low level implementation. They are as platform independent as possible. Nevertheless, they need some auxiliary libraries.
-Files with .mexw32 extension: they are precompiled versions of the MEX files for use in Matlab under 32 bit Windows.
-Files with .mexw32 extension: they are precompiled versions of the MEX files for use in Matlab under 64 bit Windows.

Any suggestions and bug reports will be welcome.


To run the demo, please type at the Matlab prompt:
>> DemoHDIR

Notes:

This image restoration method is specifically designed to cope with significant amounts of both Gaussian and uniform impulsive noise. However, if you want to test it with other kinds of noise, such as impulse-only noise, you can try to change the parameters of IRN and steering kernel regression in HDIR.m.
Sites to download related methods, as of 13 July 2010:
ISKR: http://users.soe.ucsc.edu/~htakeda/KernelToolBox.htm
IRN: http://numipad.sf.net
