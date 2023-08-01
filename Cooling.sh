#!/bin/bash
srun --time=0:30:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=3000 --partition=gll_usr_gpuprod -A tra20_cpolimi --preserve-env /bin/bash << EoF
echo "Job started at " `date`
source Cooling.x
wait
echo "Job finished at " `date`
exit
EoF
