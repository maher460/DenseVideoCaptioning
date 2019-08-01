import os, sys
import re

vidfoldername = "../UCF101_mnet/"
out_path = "../UCF101_mnet_images/"

imlist=os.listdir(vidfoldername)

for infile in imlist:
	org = infile
	infile = re.sub(r'_g\d+_c\d+.avi', '', infile)
	infile = re.sub(r'v_', '', infile)
	outdir = out_path+infile+"/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	os.rename(vidfoldername+org, outdir+org)