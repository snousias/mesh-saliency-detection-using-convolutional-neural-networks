import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from commonReadOBJPointCloud import *
import scipy.io
from tensorflow.contrib.factorization import KMeans
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import hilbertcurve.hilbertcurve.hilbertcurve as hb
import pickle
import glob
import scipy.sparse as sp
from robust_pca import R_pca
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns
# ========= Generic configurations =========#
print(80 * "=")
print('Initialize')
print(80 * "=")
rootdir='./'
modelsDir='scanned/'
dataDir='data/'
fullModelPath="./scanned/armchair.obj"
patchSide=32
numOfElements = patchSide * patchSide
numberOfClasses=4
saliencyDivisions=64
useGuided = False
doRotate = True
doReadOBJ = True
rpcaneighbours=20
pointcloudnn=8
mode = "MESH"
saliencyGroundTrouthData = '_saliencyValues_of_cendroids.csv'
patchSizeGuided = numOfElements

# ========= Read models =========#
print(80 * "=")
print('Read model data')
print(80 * "=")
(path, file) = os.path.split(fullModelPath)
filename, file_extension = os.path.splitext(file)
modelName=filename
mModelSrc = rootdir +modelsDir+ modelName + '.obj'
print(modelName)

if mode == "MESH":
    mModel = loadObj(mModelSrc)
    keyPredict = 'model_mesh' + modelName
    updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided,computeDeltas=False,computeAdjacency=False,computeVertexNormals=False)




VertexNormals = []
eigenvals=[]
print(80 * "=")
print('Spectral saliency')
print(80 * "=")
iLen = len(mModel.faces)
for f_ind, f in enumerate(mModel.faces):
    if f_ind % 2000 == 0:
        print('Extract patch information : ' + str(
            np.round((100 *f_ind / iLen), decimals=2)) + ' ' + '%')
    VertexNormalsLine = np.empty(shape=[0, 3])
    patchFaces, rings = neighboursByFace(mModel, f_ind, rpcaneighbours)
    for j in patchFaces:
        VertexNormalsLine = np.append(VertexNormalsLine, [mModel.faces[j].faceNormal], axis=0)
    nn=np.asarray(VertexNormalsLine)
    conv1=np.matmul(np.transpose(nn),nn)
    w, v = LA.eig(conv1)
    val=1/np.linalg.norm(w)
    eigenvals.append(val)
    VertexNormalsLine = VertexNormalsLine.ravel()
    VertexNormals.append(VertexNormalsLine)
print(80 * "=")
print('Geometric saliency')
print(80 * "=")
VertexNormals = np.asarray(VertexNormals)
lmbda = 1 / np.sqrt(np.max(VertexNormals.shape))
mu = 10 * lmbda
rpca = R_pca(VertexNormals, lmbda=lmbda, mu=mu)
LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
print("RPCA Shape:" + str(np.shape(VertexNormals)) + "," + "Matrix Rank:" + str(
    np.linalg.matrix_rank(VertexNormals)))
print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
    np.max(LD)))
print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
    np.max(SD)))

# ========= Read models =========#

CurvatureComponent=np.asarray(eigenvals)
RPCAComponent=np.sum(np.abs(SD)**2,axis=-1)**(1./2)

print(np.shape(RPCAComponent))
print(80 * "=")
print('Combine')
print(80 * "=")
S1=(RPCAComponent - RPCAComponent.min()) / (RPCAComponent.max() - RPCAComponent.min())
E1=(CurvatureComponent - CurvatureComponent.min()) / (CurvatureComponent.max() - CurvatureComponent.min())
saliencyPerFace=(S1+E1)/2
saliencyPerVertex=np.zeros((len(mModel.vertices)))
for mVertexIndex,mVertex in enumerate(mModel.vertices):
    umbrella=[saliencyPerFace[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
    umbrella=np.asarray(umbrella)
    saliencyPerVertex[mVertexIndex]=np.max(umbrella)

#---- write to file ------
saliencyPerFace=saliencyPerFace/np.max(saliencyPerFace)
np.savetxt(rootdir + modelsDir + modelName + saliencyGroundTrouthData, saliencyPerFace, delimiter=',',fmt='%10.3f')



# step = (1 / numberOfClasses)
# saliencyValueClass = (np.floor((saliencyPerVertex / step))).astype(int)
# saliencyValueClass = np.clip(saliencyValueClass, a_min = 0, a_max = 3)
# saliencyPerVertex = saliencyValueClass * 0.33 # For visualization purposes only


# --- color models ------
for i, v in enumerate(mModel.vertices):
    # h=-((resultPerVertex[i] * 240*0.25))
    h=0
    # s=1.0
    s=0
    # v=1.0
    v=saliencyPerVertex[i]
    r, b, g = hsv2rgb(h, s, v)
    mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
exportObj(mModel, rootdir + modelsDir + modelName + "_gt_test" + ".obj", color=True)



