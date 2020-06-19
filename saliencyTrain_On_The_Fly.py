import tensorflow as tf
import numpy as np
from numpy import genfromtxt
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
import saliencyDatasetClass as dset
from saliencyCNN import *
from matplotlib import pyplot as plt
print("Import complete")
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import time
import numpy as np
import os
def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# ========= Collections =======================================================================================

Geometry = collections.namedtuple("Geometry", "vertices, normals, faces, edges, adjacency")
Vertex = collections.namedtuple("Vertex",
                                "index,position,normal,neighbouringFaceIndices,neighbouringVerticesIndices,rotationAxis,theta,color")
Face = collections.namedtuple("Face",
                              "index,centroid,vertices,verticesIndices,verticesIndicesSorted,faceNormal,area,edgeIndices,neighbouringFaceIndices,guidedNormal,rotationAxis,theta,color")
Edge = collections.namedtuple("Edge", "index,vertices,verticesIndices,length,facesIndices,edgeNormal")

# =========Generic_configurations=======================================================================================
patchSide=32
numOfElements = patchSide * patchSide
numberOfClasses=4
selectedModel=0
saliencyDivisions=64
pointcloudnn=8
rate=0.0

# =========Actions========================#
reshapeFunction='hilbert'
#simple
#hilbert
batchNormalized=''
#'normalized'
#''
if batchNormalized=='normalized':
    saliencyGroundTrouthData='_saliencyValues_of_batch_cendroids.csv'
if batchNormalized=='':
    saliencyGroundTrouthData = '_saliencyValues_of_cendroids.csv'
reshapeFunction=reshapeFunction+batchNormalized
# distType='cosine'
# distType = 'squared_euclidean'
# rootdir = 'E:/_Groundwork/FastSaliency/'
# rootdir = 'F:/_Groundwork/FastSaliency/'
rootdir='E:/Dropbox/_GroundWork/Mesh_Saliency_Detection_Using_Convolutional_Neural_Networks/'
patchSizeGuided = numOfElements
type='discrete'




mode="MESH"
if mode=="PC":
    keyTrain = '_saliency' + '_32_models_point_cloud' + 'reshaping_' + reshapeFunction + type+'_' +  str(numberOfClasses)
if mode=="MESH":
    keyTrain = '_saliency' + '_32_models_mesh' + 'reshaping_' + reshapeFunction + type+'_' + str(numberOfClasses)
modelsDir='scanned/'
dataDir='data/'
sessionsDir='sessions/Saliency/'+type+'/'+reshapeFunction+'/'
sessionPath=sessionsDir+ keyTrain
#Read Models============================================================================================================
#Read Models============================================================================================================
#Read Models============================================================================================================
#Read Models============================================================================================================
trainSetIndices=[selectedModel]
g=glob.glob(rootdir+modelsDir+'*.obj')
trainModels=[]
for i in range(0,len(g)):
    (path, file)  = os.path.split(g[i])
    filename, file_extension = os.path.splitext(file)
    filenameParts=filename.split("_")
    trainModels.append(filenameParts[0])
trainModels = (list(set(trainModels)))
trainModels.sort()
#Hilbert Curve==========================================================================================================
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
# Rotation ==========================================================================================================
target = np.asarray([0.0, 1.0, 0.0])
# Training initializations ==========================================================================================================
labels = []
saliencyValues=[]
train_data=[]
train_labels=[]
# Extracting train data =======================================================================
t = time.time()
print('Extracting train data')
for mIndex in trainSetIndices:
    # ======Read model==============================================================================
    modelName = trainModels[mIndex]
    mModelSrc = rootdir + modelsDir + modelName + '.obj'
    print(modelName)
    print('Initialize, read model', time.time() - t)
    mModel = []
    # mModel = loadObjPC(mModelSrc, nn=pointcloudnn)
    # V, inds = computePointCloudNormals(mModel, pointcloudnn)
    # exportPLYPC(mModel, rootdir + modelsDir + modelName + '_pcnorm.ply')
    mModel = loadObj(mModelSrc)
    updateGeometryAttibutes(mModel, useGuided=False, numOfFacesForGuided=patchSizeGuided)
    # exportPLYPC(mModel, rootdir + modelsDir + modelName + '.ply')

    # Read saliency ground truth ==============================================================================
    saliencyValuePath = rootdir + modelsDir + modelName + saliencyGroundTrouthData
    saliencyValue = genfromtxt(saliencyValuePath, delimiter=',')
    print('Saliency ground truth data :', saliencyValuePath)


    if type == 'discrete':
        #Discrete

        step = (1 / numberOfClasses)
        saliencyValueClass = (np.floor((saliencyValue / step))).astype(int)
        saliencyValueClass = np.clip(saliencyValueClass, a_min=0, a_max=3)

        saliencyValueClass = saliencyValueClass.tolist()
        for i in range(len(saliencyValueClass)):
            A = np.zeros(numberOfClasses, dtype=int)
            A[int(saliencyValueClass[i])] = 1
            labels.append(A)

    #+0.5

    # Fill patches =====================================================================================================

    patches = []
    # train_data = np.empty([len(mModel.faces), patchSide, patchSide, 3])
    patches=[neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]
    # for i in range(0, iLen):
    #     p, r = neighboursByFace(mModel, i, numOfElements)
    #     patches.append(p)
    #     # if mode == "PC":
    #     #     p, r = neighboursByVertex(mModel, i, numOfElements)
    #     # if mode == "MESH":
    print('Initial model complete', time.time() - t)

    iLen=len(patches)
    # Rotation and train data formulation===============================================================================
    for i, p in enumerate(patches):
        #print(i)
        #print(p)
        if i % 2000 == 0:
            print('Extract patch information : ' + str(np.round((100 * i / iLen), decimals=2)) + ' ' + '%')

        patchFacesOriginal = [mModel.faces[i] for i in p]
        normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])

        # vec = np.mean(np.asarray(
        #         [fnm.faceNormal for fnm in [mModel.faces[j] for j in neighboursByFace(mModel, i, 4)[0]]]
        #     ), axis=0)
        vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
        # vec = mModel.faces[i].faceNormal
        vec = vec / np.linalg.norm(vec)
        axis, theta = computeRotation(vec, target)
        normalsPatchFacesOriginal = rotatePatch(normalsPatchFacesOriginal, axis, theta)
        normalsPatchFacesOriginalR = normalsPatchFacesOriginal.reshape((patchSide, patchSide, 3))
        if reshapeFunction == "hilbert":
            for hci in range(np.shape(I2HC)[0]):
                normalsPatchFacesOriginalR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatchFacesOriginal[:,
                                                                            HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
        train_data.append((normalsPatchFacesOriginalR + 1.0 * np.ones(np.shape(normalsPatchFacesOriginalR))) / 2.0)




# Dataset and labels summarization ========================================================================
train_data = np.asarray(train_data)


if type == 'discrete':
    train_labels = np.asarray(labels)



# Training params
final_iter = 25000
# Assign the batch value
batch_size = 600
# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = patchSide
num_channels = 3
num_classes = numberOfClasses
data = dset.read_train_sets(train_data, img_size, train_labels, validation_size=validation_size)
# Display the stats
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


# Architecture =================================================================================
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.333,
    allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

x = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
filter_size_conv1 = 3
num_filters_conv1 = 32
filter_size_conv2 = 3
num_filters_conv2 = 64
filter_size_conv3 = 3
num_filters_conv3 = 128
filter_size_conv4 = 3
num_filters_conv4 = 256
fc_layer_size = 128
# Create all the layers
layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1,
                                         maxpool=True
                                         # ,_padding='SAME',
                                         # ,maxpool=True
                                         )
layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2,
                                         maxpool=True
                                         )
layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3,
                                         maxpool=True
                                         )
layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                         num_input_channels=num_filters_conv3,
                                         conv_filter_size=filter_size_conv4,
                                         num_filters=num_filters_conv4)
layer_flat = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True,
                            dropoutRate=rate)
layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=fc_layer_size,
                            use_relu=True,
                            dropoutRate=rate)
layer_fc3 = create_fc_layer(input=layer_fc2,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False,
                            dropoutRate=rate)


if type == 'discrete':
    y_pred = tf.nn.softmax(layer_fc3, name='y_pred')
    y_pred_cls = tf.argmax(y_pred, axis=1)
    session.run(tf.global_variables_initializer())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                               labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())


# Training ============================================================================================================
# Display all stats for every epoch
def show_progress(epoch, acc, val_acc, val_loss, total_epochs):
    msg = "Training Epoch {0}/{4} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss, total_epochs))

if type == 'discrete':
    total_iterations = 0
    saver = tf.train.Saver()
    print("")
    trainAcc = []
    valAcc = []
    # Train
    num_iteration = final_iter
    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch = data.valid.next_batch(batch_size)
        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            acc = session.run(accuracy, feed_dict=feed_dict_tr)
            val_acc = session.run(accuracy, feed_dict=feed_dict_val)
            trainAcc.append(acc)
            valAcc.append(val_acc)
            total_epochs = int(num_iteration / int(data.train.num_examples / batch_size)) + 1
            show_progress(epoch, acc, val_acc, val_loss, total_epochs)
            saver.save(session, rootdir + sessionPath)
    total_iterations += num_iteration
    plt.plot(np.asarray(range(0, len(trainAcc))), trainAcc)
    plt.show()
    plt.plot(np.asarray(range(0, len(valAcc))), valAcc)
    plt.show()
    np.savetxt(rootdir + dataDir + keyTrain + '_trainAcc.csv', trainAcc, delimiter=",", fmt='%.3f')
    np.savetxt(rootdir + dataDir + keyTrain + '_valAcc.csv', valAcc, delimiter=",", fmt='%.3f')




# Calculate execution time
dur = time.time() - t
print("")
if dur < 60:
    print("Execution Time:", dur, "seconds")
elif dur > 60 and dur < 3600:
    dur = dur / 60
    print("Execution Time:", dur, "minutes")
else:
    dur = dur / (60 * 60)
    print("Execution Time:", dur, "hours")