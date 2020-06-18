import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from CVAEplot import *
from commonReadOBJColorV2 import *
import hilbertcurve.hilbertcurve as hb




# # distType='cosine'
# distType = 'squared_euclidean'
# rootdir= 'E:/_Groundwork/FastMeshDenoising/'
# root = rootdir+'Saliency/'
# mIndex=2
# trainModels = [
#     'fandisk',
#     'cad',
#     'casting',
#     'part_Lp',
#     'pyramid'
#     ]
#
# modelName = trainModels[mIndex]
# mModelSrc = root + modelName + '.obj'
# print(modelName)
# mModel = []
# mModel = loadObj(mModelSrc)
# updateGeometryAttibutes(mModel)








p = 32
N = 2
hilbert_curve = hb.HilbertCurve(p, N)
for ii in range(p*p):
    print('coords(h={},p={},N={}) = {}'.format(
        ii, p, N, hilbert_curve.coordinates_from_distance(ii)))


# # due to the magic of arbitrarily large integers in
# # Python (https://www.python.org/dev/peps/pep-0237/)
# # these calculations can be done with absurd numbers
# p = 512
# N = 10
# hilbert_curve = HilbertCurve(p, N)
# ii = 123456789101112131415161718192021222324252627282930
# coords = hilbert_curve.coordinates_from_distance(ii)
# print('coords(h={},p={},N={}) = {}'.format(ii, p, N, coords))



# from hilbertcurve.hilbertcurve.hilbertcurve import HilbertCurve
#
# p = 2
# N = 2
# hilbert_curve = HilbertCurve(p, N)
# for ii in range(4):
#     print('coords(h={},p={},N={}) = {}'.format(
#         ii, p, N, hilbert_curve.coordinates_from_distance(ii)))