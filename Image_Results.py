import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

No_of_Dataset = 3


def Sample_Images():
    for n in range(No_of_Dataset):
        Orig = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        label = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        class_1 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
        class_2 = ['Angry', 'Contempt', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        class_3 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
        Classes = [class_1, class_2, class_3]
        if label.shape[1] > 5:
            label = label[:, :5]
        for i in range(label.shape[1]):
            tar = label[:, i]
            ind1 = np.where(tar == 1)[0]
            image = Orig[ind1]
            ind = [1, 2, 4, 5, 6, 7]
            fig, ax = plt.subplots(2, 3)
            plt.suptitle(Classes[n][i] + " Sample Images from Dataset " + str(n + 1))
            plt.subplot(2, 3, 1)
            plt.title('Image-1')
            plt.imshow(image[ind[0]])
            plt.subplot(2, 3, 2)
            plt.title('Image-2')
            plt.imshow(image[ind[1]])
            plt.subplot(2, 3, 3)
            plt.title('Image-3')
            plt.imshow(image[ind[2]])
            plt.subplot(2, 3, 4)
            plt.title('Image-4')
            plt.imshow(image[ind[3]])
            plt.subplot(2, 3, 5)
            plt.title('Image-5')
            plt.imshow(image[ind[4]])
            plt.subplot(2, 3, 6)
            plt.title('Image-6')
            plt.imshow(image[ind[5]])
            plt.show()


if __name__ == '__main__':
    Sample_Images()
