import pickle

from sklearn.cluster import KMeans
import numpy as np
from color_model import read_image_from_filelist
import os
import cv2
import shutil
import imutils
import json
import joblib

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SceneModel():
    def __init__(self):
        self.kmeans = None

    def __reshape_data(self, X):

        n, nx, ny = X.shape
        X = X.reshape(n, nx * ny)

        return X

    def __save_result(self, title, L, filelist):

        for i in range(self.kmeans.cluster_centers_.shape[0]):
            dname = f'data/kmeans-{title}-class_{i}'

            if not os.path.exists(dname):
                os.mkdir(dname)
            # os.removedirs(dname)

            idx = np.where(L == i)[0]
            for j in idx:
                fn = os.path.basename(filelist[j])
                f, e = os.path.splitext(fn)
                nf = f + '-grayscale' + e
                path = os.path.join(dname, nf)
                # cv2.imwrite(path, train_images[j].astype('uint8'))
                shutil.copy(filelist[j], dname)

    def learn(self, train_filelist, mean_filelist=None):
        nclusters=8
        mfeatures="k-means++"

        print('Reading images ...')
        if mean_filelist:
            mean_images, _ = read_image_from_filelist(mean_filelist)
            mean_features = self.__reshape_data(np.array(self.extract_features(mean_images)))
            nclusters=mean_features.shape[0]
            mfeatures=mean_features

        train_images, train_filepath = read_image_from_filelist(train_filelist)
        train_features = self.__reshape_data(np.array(self.extract_features(train_images)))

        print('Learning ...')
        self.kmeans = KMeans(n_clusters=nclusters,
                             init=mfeatures,
                             random_state=0,
                             max_iter=1000).fit(train_features)

        print('Saving ...')
        L = self.kmeans.labels_
        self.__save_result('train', L, train_filepath)

        print('Done.')

        return L

    def test(self, test_filelist):

        # mean_images=read_image_from_filelist(mean_filelist)
        test_images, test_filepath = read_image_from_filelist(test_filelist)
        # mean_features=self.extract_feature(mean_images)

        test_features = self.__reshape_data(np.array(self.extract_features(test_images)))

        self.kmeans.predict(test_features)
        L = self.kmeans.labels_
        self.__save_result('test', L, test_filepath)

    def classify(self, frame):
        feature=self.extract_feature(frame)
        label=self.kmeans.predict(feature)
        return label[0]

    def extract_features(self, frames):
        features = []
        for f in frames:
            features.append(self.extract_feature(f))
        return features

    def extract_feature(self, frame):
        frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame2 = imutils.resize(frame2, int(frame.shape[0]*0.5))
        frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
        # cv2.equalizeHist(frame2)
        hist = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        # hist.resize(hist.size)
        hist /= hist.sum()
        hist=hist[128:]
        return np.transpose(hist)

    def load_model(self, filename):
        # with open(filename, 'r') as fd:
            # data=json.load(fd)
            # self.kmeans = KMeans(n_clusters=data['nclusters'],
            #                      init=data['cluster_centers'],
            #                      n_init=1)
            # self.kmeans.fit(data['cluster_centers'])
            # self.kmeans=pickle.load(fd)

        self.kmeans=joblib.load(filename)

    def save_model(self, filename):
        if not self.kmeans:
            return False

        # centers=self.kmeans.cluster_centers_
        # with open(filename, 'w+') as fd:
            # json.dump({
            #     'nclusters':self.kmeans.cluster_centers_.shape[0],
            #     'cluster_centers': self.kmeans.cluster_centers_,
            #     'labels': self.kmeans.labels_}, fd, cls=NumpyEncoder)

            # json.dump(centers, fd)
            # pickle.dump(self.kmeans, fd)
        joblib.dump(self.kmeans, filename)


if __name__ == '__main__':
    scenemodel = SceneModel()
    # scenemodel.learn('mean_filelist.txt', 'train_filelist.txt')
    Ltr=scenemodel.learn('data/ccd.txt', None)
    # scenemodel.test('data/ccd.txt')
    scenemodel.save_model('scenemodel.joblib')

    scenemodel.load_model('scenemodel.joblib')

    Lte=[]
    images, filenames=read_image_from_filelist('data/ccd.txt')
    for image in images:
        label=scenemodel.classify(image)
        # print(label)
        Lte.append(label)

    r=np.array(Lte)==Ltr
    print(100 * np.where(r == True)[0].size / float(len(Lte)))