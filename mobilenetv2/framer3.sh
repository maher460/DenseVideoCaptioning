for d in */ ; do
    for i in "$d"*.avi; do
    	ffmpeg -i "$i" -vf fps=1 "${i%.*}"_thumb%06d.jpg -hide_banner;
    done
done