import os, sys
from PIL import Image
import re
size = 32
vidfoldername = "./yt_out2/" #"VID_ehGHCYKzyZ8_thumb"
out_path = "./yt_resized/"
imlist=os.listdir(vidfoldername)

for infile in imlist:
	if ".jpg" in infile:
		outfile = vidfoldername + infile
		img = Image.open(outfile)
		new_img = img.resize((size,size))
		outdir = out_path+"/"+re.sub(r'_thumb.+', '', infile)
		if not os.path.exists(outdir):
			os.makedirs(outdir)
		new_img.save(outdir+"/"+infile, "JPEG", optimize=True)