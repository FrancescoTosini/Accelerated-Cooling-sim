module load gnu
rm -f Cooling.exe
gcc -pg -o Cooling.exe -O3 Cooling.c  -lm
rm -f FieldValues0???.ppm FieldValues0???.jpg
./Cooling.exe
