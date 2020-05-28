import shutil

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import random
from scipy.stats import skewnorm, norm, expon
import random


# from sdetector import VideoFrame


def read_image(filepath):
    img = cv2.imread(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# test
# data=norm.rvs(10.0, 2.5, size=500)
# show_distribution(data, norm.pdf, *(10.0, 2.5))
# data=skewnorm.rvs(10.0, 2.5, size=500)
# show_distribution(data, skewnorm.pdf, *(10.0, 2.5))
def show_distribution(data, pdf, color, *args):
    plt.hist(data, bins=100, density=True, alpha=0.6, color=color)

    n = len(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, n)
    p = pdf(x, *args)
    plt.plot(x, p, 'k', linewidth=2)
    # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    # plt.title(title)


'''
array([[[63, 54, 37],
        [54, 45, 28]],

       [[50, 41, 24],
        [67, 58, 41]]], dtype=uint8)

np.diff(ccdf[0:2, 0:2].astype(np.int16))
array([[[ -9, -17],
        [ -9, -17]],

       [[ -9, -17],
        [ -9, -17]]], dtype=int16)

np.sum(np.diff(ccdf[0:2, 0:2].astype(np.int16)), axis=2).flatten()
array([-26, -26, -26, -26])
'''

'''
    pwd : ~/Worksapce/sdetector
    source : ./data/act1/frame-000.jpg
    dest : ./data/classification-dolor
    
    make_symlink(source, dest)
        
    ./data/classification-color/frame-000.jpg 
        points 
    ./data/act1/frame-000.jpg
'''


def make_symlink(filepath, dname):
    fn = os.path.basename(filepath)
    f, e = os.path.splitext(fn)
    nf = f + '-grayscale' + e
    path = os.path.join(dname, nf)
    # cv2.imwrite(path, train_images[j].astype('uint8'))
    # shutil.copy(filelist[j], dname)
    s = os.path.relpath(filepath, dname)
    d = os.path.join(dname, fn)
    try:
        os.symlink(s, d)
    except OSError as e:
        return False

    return True


def get_color_variance(img, nsamples):
    w = img.shape[1]
    h = img.shape[0]

    widx = np.random.randint(w, size=nsamples)
    hidx = np.random.randint(h, size=nsamples)

    return np.sum(np.diff(img[hidx, widx, :].astype(np.int16)), axis=1).flatten()


def is_color(frame, nsamples):
    fr = frame
    stat = get_color_variance(fr, nsamples)
    r = float(len(list(filter(lambda x: x < 5, stat)))) / nsamples
    is_ir = r < 0.95
    return is_ir


def read_filelist(filename):
    filelist = []
    with open(filename) as fd:
        for line in fd.readlines():
            filename = line.strip()
            if len(filename) < 1:
                continue

            if not os.path.exists(filename):
                print(f'{filename} : not exists')
                continue

            filelist.append(filename)

    return filelist


def read_image_from_filelist(filename):
    images = []
    filelist = read_filelist(filename)
    for filename in filelist:
        img = read_image(filename)
        images.append(img)

    return images, filelist


def read_images(filelist):
    data = []
    images = read_image_from_filelist(filelist)
    data = list(map(lambda x: get_color_variance(x, 200), images))

    return data


def classify_color():
    resultdir = 'data/classification-ccd'
    if os.path.exists(resultdir):
        shutil.rmtree(resultdir)
    os.mkdir(resultdir)

    filename = 'data/filelist-all.txt'
    # filename = 'data/ccd.txt'
    filelist = read_filelist(filename)
    for i, f in enumerate(filelist):
        frame = read_image(f)
        if is_color(frame, 100):
            # dstfile = os.path.join(resultdir, '../../'+os.path.basename(f))
            # dstfile = '../' + os.path.basename(f)
            # os.symlink(dstfile, f)
            # f2=f.replace('/', '\\')
            # shutil.copy(f, resultdir)
            # print(f'{i:03d}/{len(filelist)} : {f}')
            make_symlink(f, resultdir)
            print(f'{f}')


# filelist: './ccd.txt' or './ir.txt'
def learn(filelist, show_dist=True):
    data = read_images(filelist)

    location, scale = norm.fit(data)
    print(f'normal distribution : location={location:.2f} scale={scale:.2f}')
    if show_dist:
        show_distribution(data, norm.pdf, 'r', *(location, scale))

    # location, scale=expon.fit(data, floc=0)
    # print(f'exponential distribution : location={location:.2f} scale={scale:.2f}')
    # if show_dist:
    #     show_distribution(data, expon.pdf, 'g', *(location, 1/scale))

    shape, location, scale = skewnorm.fit(data)
    print(f'skewed normal distribution : shape={shape:.2f} location={location:.2f} scale={scale:.2f}')
    if show_dist:
        show_distribution(data, skewnorm.pdf, 'b', *(shape, location, scale))


if __name__ == '__main__':
    # ccd : color 컴포넌트 variation 이 크다
    # ir :  color 컴포넌트 variation 이 작다

    # 픽셀의 채널 variation?
    # 전체 이미지에서 픽셀의 variation?

    # histogram equalization
    # ccd_file='./image/20200514/images/P20051421541510-CCD.jpg'
    # ir_file='./image/20200514/images/P20051421473810.jpg'

    # ccd_img=read_image(ccd_file)
    # ir_img=read_image(ir_file)

    # plt.subplot(1, 2, 1)
    # plt.imshow(ccd_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(ir_img)
    # plt.show()
    # plt.subplot(1, 2, 1)
    # plt.title('ir')
    # learn('./ir.txt', True)

    # plt.subplot(1, 2, 2)
    # plt.title('ccd')
    # learn('./ccd.txt', True)

    classify_color()

    plt.show()
