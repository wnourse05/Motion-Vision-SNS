import pickle
import os
import cv2
import numpy as np

def process_file(file_path, index, num_frames, scaled_width, scaled_height):
    # Load the trial data
    data = pickle.load(open(file_path, 'rb'))
    print(index)

    # If velocity is positive, skip
    vel = data['cmd_vel'][0]
    if vel > -0.1:  # 0.05 is too slow, stutters
        print('Wrong velocity')
        return None, None#, None

    images = data['image']  # set of 90 images
    # Skip trials where there is no change over the course of the trial
    no_change = all(np.array_equal(i, images[0]) for i in images)
    if no_change:
        print('No change in image data')
        return None, None#, None

    # left_frames = []
    # right_frames = []
    frames = np.zeros([1, num_frames, scaled_height, scaled_width])

    for i in range(num_frames):
        print('%i    Frame %i/%i' % (index, i + 1, num_frames))
        img = data['image'][i]  # grab frame

        # Resize the image using the fastest interpolation method (cv2.INTER_NEAREST)
        scaled_image = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        scaled_image_green = scaled_image[:, :, 1]  # isolate green channel
        # store scaled frame in buffers
        # left_frames.append(scaled_image_green[:, :32])
        # right_frames.append(scaled_image_green[:, 32:])
        frames[0,i,:,:] = scaled_image_green

    return frames, vel

def process_files_in_directory(directory_path):
    # List all files in the directory
    file_list = os.listdir(directory_path)
    num_files = len(file_list)
    num_frames = 90  # number of frames
    num_frames_trial = 30
    scaled_width, scaled_height = 64, 24  # size of one eye
    data = None
    labels = None

    # Iterate through each file and perform the operation
    for i in range(num_files):  # Fix the loop range
        file_path = os.path.join(directory_path, file_list[i])  # get path
        frames, vel = process_file(file_path, i, num_frames, scaled_width, scaled_height)  # get trial
        if vel is not None:
            trial0 = np.zeros([1,num_frames_trial,scaled_height,scaled_width])
            trial1 = np.zeros([1,num_frames_trial,scaled_height,scaled_width])
            trial2 = np.zeros([1,num_frames_trial,scaled_height,scaled_width])
            trial0[0,:,:,:] = frames[0, :num_frames_trial,:,:]
            trial1[0,:,:,:] = frames[0, num_frames_trial:2*num_frames_trial, :, :]
            trial2[0,:,:,:] = frames[0, 2*num_frames_trial:, :, :]
            if data is None:
                data = np.concatenate([trial0, trial1, trial2], axis=0)#, axis=3)
                labels = np.array([vel, vel, vel])
            else:
                # data0 = np.array(left)[:, :, :, np.newaxis]  # Add a new axis for stacking
                # data_right = np.array(right)[:, :, :, np.newaxis]  # Add a new axis for stacking
                # data = np.concatenate([data, data_left, data_right], axis=0)
                data = np.concatenate([data, trial0, trial1, trial2], axis=0)
                labels = np.concatenate([labels, np.array([vel, vel, vel])])

    dataset = {'data': data, 'labels': labels}
    pickle.dump(dataset, open('dataset.p', 'wb'))

directory_path = '../FlyWheelTrials/'
process_files_in_directory(directory_path)