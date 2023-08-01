In the 07-Cooling directory there is a sequential source code to be optimized
and scripts for compiling and running on M100 (with few changes on any other 
Linux platforms).

Tests have been realized with GNU compilers, but other C and Fortran compilers 
should do as well.

The Xdots and Ydots parameters may be changed to 1000 in developing the optimised 
version, as well as reducing the "Computed steps" at the end of the 'Cooling.inp' 
input file, but the final tests and benchmarks should be carried out with the original
values (1400 and 240 respectively).

To check the correctness of the program, the values printed at all the steps by the 
original and the optimised program versions should match, although small differences 
could be acceptable. Also the images generated at the last stages should look similar.

The generated image files could be assembled to create a video like FieldValues.avi,
10 frames/sec.

Would you please read the comments in the source code for further instructions