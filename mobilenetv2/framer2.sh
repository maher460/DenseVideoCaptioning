for i in ../UCF101_mnet/*.avi; do
	ffmpeg -i "$i" -vf fps=1 "${i%.*}"_thumb%06d.jpg -hide_banner;
done
