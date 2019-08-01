import os, sys
from PIL import Image
import re
size = 32
vidfoldername = "../UCF101_mnet_images/" #"VID_ehGHCYKzyZ8_thumb"
out_path = "../UCF101_mnet_images_resized/"


dirs=os.listdir(vidfoldername)

for d in dirs:
	if not os.path.exists(out_path+d+"/"):
		os.makedirs(out_path+d+"/")
	imlist = os.listdir(vidfoldername+d+"/")
	for infile in imlist:
		outfile = vidfoldername+d+"/" + infile
		img = Image.open(outfile)
		new_img = img.resize((size,size))
		outdir = out_path+d+"/"
		new_img.save(outdir+infile, "JPEG", optimize=True)