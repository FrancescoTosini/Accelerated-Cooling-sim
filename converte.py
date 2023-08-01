"""
   It converts PPM file in JPEG images using ImageMagick convert function
"""

import os

import glob
ppm_l = glob.glob("*.ppm")

for nome in ppm_l:
   nome_ = nome.split(".")
   nuovo=nome_+".jpg"
   cmdexe=" ".join(["convert ",nome,nuovo])
   r=os.system(cmdexe)
   print "r, cmdexe = ",r,cmdexe   

