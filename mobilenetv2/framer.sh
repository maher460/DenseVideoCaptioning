for d in */ ; do
    for i in "$d"*.mp4; do
    	ffmpeg -i "$i" -vf "select=not(mod(n\,64))" -vsync vfr "${i%.*}"_thumb%06d.jpg -hide_banner;
    done
done