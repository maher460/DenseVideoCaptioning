import os
import glob
from shutil import copyfile
from random import shuffle

def divide_data():

	# I need to transfer data to three different folders: test, train and validation
	# test is 10%, validation is 10% and train is 80% of data
	#print('I need to start :)))')

	videos_dir = '../UCF101_mnet_images_resized/'

	train_dir = '../UCF101_mnet_images_resized_train/'
	test_dir = '../UCF101_mnet_images_resized_test/'
	validation_dir = '../UCF101_mnet_images_resized_val/'

	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
	if not os.path.exists(validation_dir):
		os.makedirs(validation_dir)


	# first we need to iterate over the videos_dir and transfer 10% of videos to validation_dir, 10% to test_dir and rest
	# to train  

	video_folders = [os.path.join(videos_dir , o) for o in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir , o))]

	for video_folder in video_folders:
		folder_name = video_folder.split('/')[-1]
		print(folder_name)
		vid_paths = glob.glob(video_folder + '/*.jpg')	  
			
		number_of_videos = len(vid_paths)       

		print('number of videos is ' + str(number_of_videos))

		number_of_test_videos = number_of_videos / 10
	
		print('number of test videos is ' + str(number_of_test_videos))


		train_folder_path = os.path.join(train_dir , folder_name)
		test_folder_path = os.path.join(test_dir , folder_name)
		validation_folder_path = os.path.join(validation_dir , folder_name)

		if not os.path.exists(train_folder_path):
			os.makedirs(train_folder_path)

		if not os.path.exists(test_folder_path):
			os.makedirs(test_folder_path)

		if not os.path.exists(validation_folder_path):
			os.makedirs(validation_folder_path) 

		rand_indices = range(len(vid_paths))
		shuffle(rand_indices)
		for i in range(len(vid_paths)):
			idx = rand_indices[i]
			video_name = vid_paths[idx].split('/')[-1]
			print('video name is ' + video_name)
			if (i < number_of_test_videos ):
				copyfile(vid_paths[i], os.path.join(test_folder_path , video_name))

			elif ( i > number_of_test_videos - 1 ) and (i < 2*(number_of_test_videos)):
				copyfile(vid_paths[i], os.path.join(validation_folder_path , video_name))

			else:
				copyfile(vid_paths[i], os.path.join(train_folder_path , video_name))


if __name__ == '__main__':
	divide_data()