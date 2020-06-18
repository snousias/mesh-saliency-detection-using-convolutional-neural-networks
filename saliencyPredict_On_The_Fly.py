import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
#from CVAEplot import *
from commonReadOBJPointCloud import *
import scipy.io
from tensorflow.contrib.factorization import KMeans
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import pickle
import hilbertcurve.hilbertcurve.hilbertcurve as hb
import glob
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


################################################
def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# ================== Generic_configurations ==================#
rootdir='./'
patchSide=32
numOfElements = patchSide * patchSide
numberOfClasses=4

fullModelPath="./scanned/head.obj"
saliencyGroundTrouthData = '_saliencyValues_of_cendroids.csv'
saliencyDivisions=64
pointcloudnn=8
rate=0.0
#simple
#hilbert
reshapeFunction='hilbert'
#discrete
#continuous
type='discrete'
mode="MESH"
useGuided = False
groundTruthExists=False
doRotate = True
doReadOBJ = True
exportPatchFormat=True
retrieveStoredDataset=not exportPatchFormat
patchSizeGuided = numOfElements
if mode=="MESH":
    keyTrain = '_saliency' + '_32_models_mesh' + 'reshaping_' + reshapeFunction + type+'_' + str(numberOfClasses)
modelsDir='scanned/'
dataDir='data/'
sessionsDir='sessions/Saliency/'+type+'/'+reshapeFunction+'/'
sessionPath=sessionsDir+ keyTrain

print(80 * "=")
print('Initialize')
print(80 * "=")



# ================== Load stored session ==================#
sess = get_session()
#sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(rootdir+sessionPath+'.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint(rootdir+sessionsDir))
# Accessing the default graph which we have restored
graph = tf.compat.v1.get_default_graph()
# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")
## Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
start = time.time()



# ================== Read model path ================== #
(path, file) = os.path.split(fullModelPath)
filename, file_extension = os.path.splitext(file)
modelName=filename

saliencyGroundTrouthDataFile=path+'/'+modelName+saliencyGroundTrouthData
if os.path.exists(saliencyGroundTrouthDataFile):
    groundTruthExists=True
    print("Groundtruth data exist, confusion matrix will be generated")
else:
    print("Groundtruth data are missing, confusion matrix will not be generated")

if mode=="MESH":
    keyPredict = 'model_mesh'+ modelName
# ================== Struct ================== #
Geometry = collections.namedtuple("Geometry", "vertices, normals, faces, edges, adjacency")
Vertex = collections.namedtuple("Vertex",
                                "index,position,normal,neighbouringFaceIndices,neighbouringVerticesIndices,rotationAxis,theta,color")
Face = collections.namedtuple("Face",
                              "index,centroid,vertices,verticesIndices,verticesIndicesSorted,faceNormal,area,edgeIndices,neighbouringFaceIndices,guidedNormal,rotationAxis,theta,color")
Edge = collections.namedtuple("Edge", "index,vertices,verticesIndices,length,facesIndices,edgeNormal")

# ================== Hilbert curve ================== #
p = patchSide
N = 2
hilbert_curve = hb.HilbertCurve(p, N)

I2HC=np.empty((p*p,2))
HC2I=np.empty((p,p))
hCoords=[]
for ii in range(p*p):
    h=hilbert_curve.coordinates_from_distance(ii)
    hCoords.append(h)
    I2HC[ii,:]=h
    HC2I[h[0],h[1]]=ii
I2HC=I2HC.astype(int)
HC2I=HC2I.astype(int)
I = np.eye(3)

# ================== Initializations ================== #
t = time.time()
NormalsOriginal = []
labels = []
saliencyValues=[]
train_data=[]
train_labels=[]

# ================== Run inference ================== #

if True:
    print(80*"=")
    print('Extracting model data')
    print(80*"=")
    if True:
        mModelSrc = rootdir+modelsDir + modelName + '.obj'
        print(modelName)
        print('Initialize, read model', time.time() - t)
        mModel = []
        if mode == "MESH":
            mModel = loadObj(mModelSrc)
            updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=numOfElements)
            iLen = len(mModel.faces)
        predict_data = np.empty([iLen,patchSide, patchSide, 3])
        prediction=np.empty([iLen,1])
        for i in range(0, iLen):
            if i % 2000 == 0:
                print('Extract patch information : ' + str(
                    np.round((100 * i / iLen), decimals=2)) + ' ' + '%')
            if mode=="MESH":
                p,r = neighboursByFace(mModel, i, numOfElements)
                patchFacesOriginal=[mModel.faces[i] for i in p]
                normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])
                if doRotate:
                    vec = np.mean(np.asarray(
                        [fnm.faceNormal for fnm in [mModel.faces[j] for j in neighboursByFace(mModel, i, 4)[0]
                                                    ]]),
                        axis=0)
                    # vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
                    # vec = mModel.faces[i].faceNormal
                    vec = vec / np.linalg.norm(vec)
                    target = np.asarray([0.0, 1.0, 0.0])
                    axis, theta = computeRotation(vec, target)
                    normalsPatchFacesOriginal = rotatePatch(normalsPatchFacesOriginal, axis, theta)
                normalsPatchFacesOriginalR = normalsPatchFacesOriginal.reshape((patchSide, patchSide, 3))
                if reshapeFunction == "hilbert":
                    for hci in range(np.shape(I2HC)[0]):
                        normalsPatchFacesOriginalR[I2HC[hci,0], I2HC[hci,1], :] = normalsPatchFacesOriginal[:, HC2I[I2HC[hci,0],I2HC[hci,1]]]
                        #normalsPatchFacesOriginalR[hc[0], hc[1], :] = normalsPatchFacesOriginal[:, hCoords.index(hc)]
                pIn=(normalsPatchFacesOriginalR + 1.0 * np.ones(np.shape(normalsPatchFacesOriginalR))) / 2.0
                x_batch = pIn.reshape(1, patchSide, patchSide, 3)
                y_test_images = np.zeros((1,numberOfClasses)) #conf.numberOfClasses
                feed_dict_testing = {x: x_batch, y_true: y_test_images}
                result = sess.run(y_pred, feed_dict=feed_dict_testing)
                a = result[0].tolist()
                r = 0
                # Finding the maximum of all outputs
                max1 = max(a)
                index1 = a.index(max1)
                prediction[i]=index1

# ================== Per face => per vertex ================== #

result = np.asarray(prediction)
resultPerVertex=np.zeros((len(mModel.vertices)))
for mVertexIndex,mVertex in enumerate(mModel.vertices):
    umbrella=[result[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
    umbrella=np.asarray(umbrella)
    resultPerVertex[mVertexIndex]=np.max(umbrella)

# ================== Generate colored model ================== #
for i, v in enumerate(mModel.vertices):
    # h=-((resultPerVertex[i] * 240*0.25))
    h=0
    # s=1.0
    s=0
    # v=1.0
    v=(resultPerVertex[i]*0.25)
    r, b, g = hsv2rgb(h, s, v)
    mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
exportObj(mModel, rootdir + modelsDir + modelName + keyTrain + ".obj", color=True)



# ================== Validation & Confusion matrix ================== #
if groundTruthExists:
    print(80*"=")
    print("Groundtruth exists, confusion matrices will be generated")
    print(80 * "=")

    saliencyValuePath = rootdir + modelsDir + modelName + saliencyGroundTrouthData
    saliencyValue = np.genfromtxt(saliencyValuePath, delimiter=',')
    print('Saliency ground truth data :', saliencyValuePath)

    step = (1 / numberOfClasses)
    saliencyValueClass = (np.floor((saliencyValue / step) + 0.5)).astype(int)
    # saliencyValueClass = np.clip(saliencyValueClass, a_min=0, a_max=3)

    # step = (1 / numberOfClasses)
    # saliencyValueClass = (np.floor((saliencyValue / step))).astype(int)
    # saliencyValueClass = np.clip(saliencyValueClass, a_min=0, a_max=3)

    saliencyValueClass=saliencyValueClass.tolist()
    _pred=prediction
    _true=np.asarray(saliencyValueClass).astype(int).tolist()
    #print(_true)
    #print(_pred)
    classes=np.asarray(range(0,4)).astype(int).tolist()
    normalize=True
    cm =confusion_matrix(_true, _pred)
    # Only use the labels that appear in the data
    classes = unique_labels(_true, _pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    # ax.set(xticks=np.arange(cm.shape[1]),
    #            yticks=np.arange(cm.shape[0]),
    #            # ... and label them with the respective list entries
    #            xticklabels=classes, yticklabels=classes,
    #            title='',
    #            ylabel='True label',
    #            xlabel='Predicted label')
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #              rotation_mode="anchor")
    #Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()