import numpy as np
from PIL import Image


class Data:
    def __init__(self):
        self.Ntrain = 15 + 50
        self.Ntest = 5 + 10
        self.batch_size = 10
        self.img_size = (64*3) * (64*3)  # row*cols
        myroot = 'C:\\Users\\Yuval\\Pictures\\Micka\\CNN_data\\TrainGRY\\'
        imgs = np.zeros((self.Ntrain, self.img_size))
        ind = -1
        for i in range(self.Ntrain):
            try:
                img_train = Image.open(myroot + 'train'+str(i+1)+'.png')
                ind = ind+1
                print('ind=', ind)
            except:
                continue
            img_arr1 = np.array(img_train)
            imgs[[ind], :] = np.resize(img_arr1, (1, self.img_size))
        self.train_imgs = imgs.astype(dtype=np.float32)

        train_lables = np.zeros((self.Ntrain, 2))
        train_lables[0:110, 0] = 1  # self
        train_lables[110:, 1] = 1  # rest
        self.train_labels = train_lables.astype(dtype=np.float32)

        # Test Data
        imgs = np.zeros((self.Ntest, self.img_size))
        for i in range(self.Ntest):
            img_test = Image.open(myroot + 'test' + str(i + 1) + '.png')
            img_arr2 = np.array(img_test)
            img_arr2.resize(self.img_size, 1)
            imgs[[i], :] = img_arr2.transpose()
        self.test_imgs = imgs.astype(dtype=np.float32)

        test_labels = np.zeros((self.Ntest, 2))
        test_labels[0:10, 0] = 1  # self
        test_labels[10:, 1] = 1  # rest
        self.test_labels = test_labels.astype(dtype=np.float32)

    def get_batch(self, i, train):
        if train:
            batch_x = self.train_imgs[i*self.batch_size:(i+1)*self.batch_size, :]
            batch_y = self.train_labels[i*self.batch_size:(i+1)*self.batch_size, :]
            dropout = 0.75
        else:
            batch_x = self.test_imgs
            batch_y = self.test_labels
            dropout = 1.0

        return batch_x, batch_y, dropout
