#!/usr/bin/env python3
from dilation import DilationNet, predict_no_tiles
from os.path import join
from glob import glob
# from tqdm import tqdm
import numpy as np
import argparse
import os.path
import cv2

cityscape_palette = np.array([[128, 64, 128],
                            [244, 35, 232],
                            [70, 70, 70],
                            [102, 102, 156],
                            [190, 153, 153],
                            [153, 153, 153],
                            [250, 170, 30],
                            [220, 220, 0],
                            [107, 142, 35],
                            [152, 251, 152],
                            [70, 130, 180],
                            [220, 20, 60],
                            [255, 0, 0],
                            [0, 0, 142],
                            [0, 0, 70],
                            [0, 60, 100],
                            [0, 80, 100],
                            [0, 0, 230],
                            [119, 11, 32]], dtype='uint8')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    args = parser.parse_args()

    # assert args.sequence is not None, 'Please provide a correct sequence number'

    # dreyeve_dir = '/home/aba/dreyeve/data/'  # local
    # dreyeve_dir = '/gpfs/work/IscrC_DeepVD/dabati/dreyeve_semantic_seg/data/'  # cineca

    # data_dir = dreyeve_dir + '{:02d}/frames'.format(int(args.sequence))  # local
    # out_dir = dreyeve_dir + '{:02d}/semseg'.format(int(args.sequence))  # local
    # assert os.path.exists(data_dir), 'Assertion error: path {} does not exist'.format(data_dir)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #
    # image_list = glob(join(data_dir, '*.jpg'))

    # read and predict a image
    # path = '/home/josephz/ws/git/personal/josephz/experiments/2018/dreyeve/semseg/imgs_dilation/cityscapes.png'
    path = '/home/josephz/tmp/data/dr-eyeve/35/frames/0057.png'
    im = cv2.imread(path)
    image_size = im.shape
    input_shape = (3, image_size[0], image_size[1])

    # get the model
    ds = 'cityscapes'
    model = DilationNet(dataset=ds, input_shape=(3, 1452, 2292))
    # model = DilationNet(dataset=ds, input_shape=(3, 884, 1396))
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    # get intermediate model
    # output = 'conv1_1'
    output = 'conv5_3'
    # output = 'fc6'
    # output = 'final'
    # output = 'ctx_conv1_1'
    # output = 'ctx_fc1'
    # output = 'ctx_final'
    # output = 'ctx_upsample'
    if output is not None:
      out_layer = model.get_layer(output)
      assert out_layer is not None
      from keras.models import Model
      model = Model(inputs=model.input, outputs=out_layer.output)
      assert model is not None
    model.summary()

    # for img in tqdm(image_list):

    y = predict_no_tiles(im, model, ds, output=output)
    import matplotlib.pyplot as plt

    import pdb
    # pdb.set_trace()
    prediction = np.argmax(y[0], axis=0)
    prediction = prediction[0:image_size[0], 0:image_size[1]]
    color_image = cityscape_palette[prediction.ravel()].reshape(image_size)
    plt.imshow(color_image)
    plt.show()
    # y_resized = np.zeros(shape=(1, 19, 270, 480))
    #
    # for c in range(0, y.shape[1]):
    #     y_resized[0, c] = cv2.resize(y[0, c], dsize=(480, 270))
    #
    # np.savez_compressed(join(out_dir, os.path.basename(img)[0:-4]), y_resized)


