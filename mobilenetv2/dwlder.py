
# Search for youtube videos
# Create a folder in data with the search query as name
# download all the videos there


import urllib
import re

from pytube import YouTube

import os

import time

print("\tDownloading youtube videos:")
count_b = 0

files = ["train_ids.txt", "val_ids.txt", "test_ids.txt"]

search_results = ["QOlSCBRmfWY","ehGHCYKzyZ8","nwznKOuZM7w","ogQozSI5V8U","nHE7u40plD0","69IsHpmRyfk","D18b2IZpxk0","pizl41xmw7k","oP77DgsbhKQ","fzp5ooc727c","uqiMw7tQ1Cc","bXdq2zI1Ms0","FsS_NCZEfaI","K6Tm5xHkJ5c","4Lu8ECLHvK4"]

search_results = []
for fn in files:
    f = open(fn, 'r')
    for line in f:
        line = line.replace("v_", "")
        line = line.replace("\n", "")
        search_results.append(line)
    f.close()

search_results = list(set(search_results))

# out_dir = "/afs/cs.pitt.edu/usr0/abj40/public/yt_downloader/yt_out"
out_dir = "/afs/cs/projects/kovashka/maher/vol2/mobilenetV2/yt_out2"

already_done = []
failed = []

if(os.path.isfile(out_dir + "/log.text")): 
    file = open(out_dir + "/log.text", 'r') 
    already_done = eval(file.read())
    search_results = list(filter(lambda x: x not in already_done, search_results))
    file.close()

total = len(search_results)

for result in search_results:

    try:

        count_b = count_b + 1

        url = "https://www.youtube.com/watch?v=" + result

        print(str("\t\t(") + str(count_b) + "/" + str(total) + "): " + result)

        yt = YouTube(url)

        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

        stream.download(output_path=out_dir, filename=result, filename_prefix="VID_")

        already_done.append(result)

        f2 = open(out_dir + "/log.text", 'w')
        f2.write(repr(already_done))
        f2.close()

    except Exception as e:
        print("\t\tERROR: Failed to download and save: " + result)
        print(e)
        failed.append(result)
        # print("\tDone downloading videos for search string: " + search_string)

f3 = open(out_dir + "/failed.text", 'w')
f3.write(repr(failed))
f3.close()