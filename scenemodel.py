from sklearn.cluster import KMeans
import numpy as np
from color_model import read_image_from_filelist, make_symlink
import os
import cv2
import shutil
import imutils
import json
import joblib
import logging
import sys
from scipy.special import kl_div


def get_coordination(roi):
    return roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SceneModel():
    def __init__(self, modeltype='kmeans'):
        self.model = None

    def __reshape_data(self, X):

        n, nx, ny = X.shape
        X = X.reshape(n, nx * ny)

        return X

    def __save_result(self, title, L, filelist):

        for i in range(self.model.cluster_centers_.shape[0]):
            dname = f'data/kmeans-{title}-class_{i}'

            if os.path.exists(dname):
                shutil.rmtree(dname)
            os.mkdir(dname)

            idx = np.where(L == i)[0]
            for j in idx:
                # fn = os.path.basename(filelist[j])
                # f, e = os.path.splitext(fn)
                # nf = f + '-grayscale' + e
                # path = os.path.join(dname, nf)
                # # cv2.imwrite(path, train_images[j].astype('uint8'))
                # # shutil.copy(filelist[j], dname)
                # s=os.path.relpath(filelist[j], dname)
                # d=os.path.join(dname, fn)
                # try:
                #     os.symlink(s, d)
                # except OSError as e:
                #     logging.info(e.args)
                make_symlink(filelist[j], dname)

    def learn_from_imagelist(self, train_images, train_filepath, mean_images=None):
        nclusters = 8
        mfeatures = "k-means++"

        logging.info('Reading images ...')
        if mean_images:
            mean_features = self.__reshape_data(np.array(self.extract_features(mean_images)))
            nclusters = mean_features.shape[0]
            mfeatures = mean_features

        train_features = self.__reshape_data(np.array(self.extract_features(train_images)))

        logging.info('Learning ...')
        self.model = KMeans(n_clusters=nclusters,
                            init=mfeatures,
                            random_state=0,
                            max_iter=1000).fit(train_features)

        logging.info('Saving ...')
        L = self.model.labels_
        self.__save_result('train', L, train_filepath)

        logging.info('Done.')

        return L

    def learn_from_filelist(self, train_filelist, mean_filelist=None):

        logging.info('Reading images ...')
        mean_images = None
        if mean_filelist:
            mean_images, _ = read_image_from_filelist(mean_filelist)

        train_images, train_filepath = read_image_from_filelist(train_filelist)
        L = self.learn_from_imagelist(train_images, train_filepath, mean_images)

        return L

    def test(self, test_filelist):

        # mean_images=read_image_from_filelist(mean_filelist)
        test_images, test_filepath = read_image_from_filelist(test_filelist)
        # mean_features=self.extract_feature(mean_images)

        test_features = self.__reshape_data(np.array(self.extract_features(test_images)))

        self.model.predict(test_features)
        L = self.model.labels_
        self.__save_result('test', L, test_filepath)

    def classify(self, frame):
        feature = self.extract_feature(frame)
        labels = self.model.predict(feature)
        return labels[0]

    def extract_features(self, frames):
        features = []
        for f in frames:
            features.append(self.extract_feature(f))
        return features

    def extract_feature(self, frame, resize_scale=1, roi=None):
        frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if resize_scale != 1:
            frame2 = imutils.resize(frame2, int(frame.shape[0] * resize_scale))
        # frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
        frame2 = cv2.medianBlur(frame2, 5)

        # cv2.equalizeHist(frame2)
        mask=None
        if roi is not None:
            mask_bg=np.ones_like(frame2)*255
            mask_bg[roi[1]:roi[3], roi[0]:roi[2]]=0
            mask=mask_bg.astype('uint8')

        hist = cv2.calcHist([frame2], [0], mask, [256], [0, 256])
        # hist.resize(hist.size)
        hist += 1
        hist /= hist.sum()
        # hist = hist[128:]
        return np.transpose(hist)

    def compare_feature(self, prev_feature, curr_feature):
        if prev_feature is None:
            return 0

        kld = np.sum(kl_div(prev_feature, curr_feature))

        return kld


    def load_model(self, filename):
        # with open(filename, 'r') as fd:
        # data=json.load(fd)
        # self.kmeans = KMeans(n_clusters=data['nclusters'],
        #                      init=data['cluster_centers'],
        #                      n_init=1)
        # self.kmeans.fit(data['cluster_centers'])
        # self.kmeans=pickle.load(fd)

        self.model = joblib.load(filename)

    def save_model(self, filename):
        if not self.model:
            return False

        # centers=self.kmeans.cluster_centers_
        # with open(filename, 'w+') as fd:
        # json.dump({
        #     'nclusters':self.kmeans.cluster_centers_.shape[0],
        #     'cluster_centers': self.kmeans.cluster_centers_,
        #     'labels': self.kmeans.labels_}, fd, cls=NumpyEncoder)

        # json.dump(centers, fd)
        # pickle.dump(self.kmeans, fd)
        joblib.dump(self.model, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # listfile = 'data/filelist-all.txt'
    listfile=sys.argv[1]
    modelfile = 'scenemodel.joblib'

    logging.info(f'Reading images ...')
    dataset, datafilelist = read_image_from_filelist(listfile)
    scenemodel = SceneModel()

    # scenemodel.learn('mean_filelist.txt', 'train_filelist.txt')
    Ltr = scenemodel.learn_from_imagelist(dataset, datafilelist)
    # scenemodel.test('data/ccd.txt')
    scenemodel.save_model(modelfile)

    scenemodel.load_model(modelfile)

    Lte = []
    images, filenames = read_image_from_filelist(listfile)
    for image in images:
        label = scenemodel.classify(image)
        # logging.info(label)
        Lte.append(label)

    r = np.array(Lte) == Ltr
    logging.info(100 * np.where(r == True)[0].size / float(len(Lte)))
