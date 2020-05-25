from sklearn.cluster import KMeans
import numpy as np
from color_model import read_image_filelist
import os
import cv2
import shutil


class SceneModel():
    def __init__(self):
        self.kmeans=None

    
    def __reshape_data(self, X):

        n, nx, ny=X.shape
        X=X.reshape(n, nx*ny)

        return X


    def __save_result(self, title, L, filelist):

        for i in range(self.kmeans.cluster_centers_.shape[0]):
            dname=f'kmeans-{title}-class_{i}'

            if not os.path.exists(dname):
                os.mkdir(dname)
            # os.removedirs(dname)

            idx=np.where(L==i)[0]
            for j in idx:
                fn=os.path.basename(filelist[j])
                f, e=os.path.splitext(fn)
                nf=f+'-grayscale'+e
                path=os.path.join(dname, nf)
                # cv2.imwrite(path, train_images[j].astype('uint8'))
                shutil.copy(filelist[j], dname)


    def learn(self, mean_filelist, train_filelist):
        mean_images, _=read_image_filelist(mean_filelist)
        train_images, train_filepath=read_image_filelist(train_filelist)
        mean_features=self.__reshape_data(np.array(self.extract_features(mean_images)))
        train_features=self.__reshape_data(np.array(self.extract_features(train_images)))
        
        self.kmeans = KMeans(n_clusters=mean_features.shape[0], 
            init=mean_features, random_state=0).fit(train_features)
        L=self.kmeans.labels_
        self.__save_result('train', L, train_filepath)


    def test(self, test_filelist):

        # mean_images=read_image_filelist(mean_filelist)
        test_images, test_filepath=read_image_filelist(test_filelist)
        # mean_features=self.extract_feature(mean_images)

        test_features=self.__reshape_data(np.array(self.extract_features(test_images)))
        
        self.kmeans.predict(test_features)
        L=self.kmeans.labels_
        self.__save_result('test', L, test_filepath)


    def classify(self, frame):
        self.kmeans.predict()


    def extract_features(self, frames):
        features=[]
        for f in frames:
            features.append(self.extract_feature(f))
        return features


    def extract_feature(self, frame):
        frame2=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hist=cv2.calcHist([frame2], [0], None, [128], [0, 256])
        return np.transpose(hist)


scenemodel=SceneModel()
scenemodel.learn('mean_filelist.txt', 'train_filelist.txt')
scenemodel.test('test_filelist.txt')