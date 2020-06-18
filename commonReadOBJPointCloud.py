import csv
import collections
import numpy as np
from sklearn.neighbors import KDTree
from scipy.sparse import dok_matrix
import time
import copy
import math
from joblib import Parallel, delayed
from numpy import linalg as LA

Geometry = collections.namedtuple("Geometry", "vertices, normals, faces, edges, adjacency")
Vertex = collections.namedtuple("Vertex",
                                "index,position,delta,normal,neighbouringFaceIndices,neighbouringVerticesIndices,rotationAxis,theta,color,edgeIndices")
Face = collections.namedtuple("Face",
                              "index,centroid,delta,vertices,verticesIndices,verticesIndicesSorted,faceNormal,area,edgeIndices,neighbouringFaceIndices,guidedNormal,rotationAxis,theta,color")
Edge = collections.namedtuple("Edge", "index,vertices,verticesIndices,length,facesIndices,edgeNormal")

def list2dict(lst):
    res_dct = {lst[i]: i for i in range(0, len(lst))}
    return res_dct


def loadObj(filename):
    vertices = []
    normals = []
    faces = []
    edges = []
    adjacency = []
    with open(filename, newline='') as f:
        flines = f.readlines()
        # Read vertices
        indexCounter = 0;
        print('Reading Vertices')
        for row in flines:
            if row[0] == 'v' and row[1] == ' ':
                line = row.rstrip()
                line = line[2:len(line)]
                coords = line.split()
                coords = list(map(float, coords))
                v = Vertex(
                    index=indexCounter,
                    position=np.asarray([coords[0], coords[1], coords[2]]),
                    delta=[],
                    normal=np.asarray([0.0, 0.0, 0.0]),
                    neighbouringFaceIndices=[],
                    neighbouringVerticesIndices=[],
                    theta=0.0,
                    edgeIndices=[],
                    rotationAxis=np.asarray([0.0, 0.0, 0.0]),
                    #color=np.asarray([coords[3], coords[4], coords[5]])
                    color = np.asarray([0.0, 0.0, 0.0])
                )
                indexCounter += 1;
                vertices.append(v)
        # Read Faces
        indexCounter = 0;
        print('Reading Faces')
        for row in flines:
            if row[0] == 'f':
                line = row.rstrip()
                line = line[2:len(line)]
                lineparts = line.strip().split()
                faceline = [];
                for fi in lineparts:
                    fi = fi.split('/')
                    faceline.append(int(fi[0]) - 1)
                f = Face(
                    index=indexCounter,
                    verticesIndices=[int(faceline[0]), int(faceline[1]), int(faceline[2])],
                    verticesIndicesSorted=[int(faceline[0]), int(faceline[1]), int(faceline[2])].sort(),
                    vertices=[],
                    centroid=np.asarray([0.0, 0.0, 0.0]),
                    delta=[],
                    faceNormal=np.asarray([0.0, 0.0, 0.0]),
                    edgeIndices=[],
                    area=0.0,
                    neighbouringFaceIndices=[],
                    guidedNormal=np.asarray([0.0, 0.0, 0.0]),
                    theta=0.0,
                    rotationAxis=np.asarray([0.0, 1.0, 0.0]),
                    color=np.asarray([0.0, 0.0, 0.0])
                )
                indexCounter += 1;
                faces.append(f)

        # Which vertices are neighbouring to each vertex
        print('Which vertices are neighbouring to each vertex')
        for idx_f, f in enumerate(faces):
            v0 = f.verticesIndices[0]
            v1 = f.verticesIndices[1]
            v2 = f.verticesIndices[2]
            if v1 not in vertices[v0].neighbouringVerticesIndices:
                vertices[v0].neighbouringVerticesIndices.append(v1)
            if v2 not in vertices[v0].neighbouringVerticesIndices:
                vertices[v0].neighbouringVerticesIndices.append(v2)
            if v0 not in vertices[v1].neighbouringVerticesIndices:
                vertices[v1].neighbouringVerticesIndices.append(v0)
            if v2 not in vertices[v1].neighbouringVerticesIndices:
                vertices[v1].neighbouringVerticesIndices.append(v2)
            if v0 not in vertices[v2].neighbouringVerticesIndices:
                vertices[v2].neighbouringVerticesIndices.append(v0)
            if v1 not in vertices[v2].neighbouringVerticesIndices:
                vertices[v2].neighbouringVerticesIndices.append(v1)

        # Which faces are neighbouring to each vertex
        print('Which faces are neighbouring to each vertex')
        for idx_f, f in enumerate(faces):
            for idx_v, v in enumerate(f.verticesIndices):
                vertices[v].neighbouringFaceIndices.append(f.index)

        print('Which faces are neighbouring to each face')
        for idx_v, v in enumerate(vertices):
            for idx_f in v.neighbouringFaceIndices:
                for jdx_f in v.neighbouringFaceIndices:
                    common = set(faces[idx_f].verticesIndices) & set(faces[jdx_f].verticesIndices)
                    if len(common) == 2:
                        faces[idx_f].neighbouringFaceIndices.append(jdx_f)

        for idx_f, fi in enumerate(faces):
            neighbouringFaceIndices=faces[idx_f].neighbouringFaceIndices
            faces[idx_f]=faces[idx_f]._replace(neighbouringFaceIndices=list(set(neighbouringFaceIndices)))

        ind_e=0
        _edges=[]
        for idx_v,vi in enumerate(vertices):
            neighbouringVerts=vertices[idx_v].neighbouringVerticesIndices
            for nv in neighbouringVerts:
                _edges.append([idx_v,nv])
        _edges = np.asarray(_edges)
        _edges.sort(axis=1)
        _edges = np.unique(_edges, axis=0)
        _edges=_edges.tolist()
        for idx_e,ei in enumerate(_edges):

            vertices[ei[0]].edgeIndices.append(idx_e)
            vertices[ei[1]].edgeIndices.append(idx_e)

            f1=vertices[ei[0]].neighbouringFaceIndices
            f2=vertices[ei[1]].neighbouringFaceIndices


            fs = list(set(f1).intersection(f2))
            E = Edge(
                index=ind_e,
                vertices=[],
                verticesIndices=[ei[0], ei[1]],
                length=0.0,
                facesIndices=fs,
                edgeNormal=np.asarray([1.0, 0.0, 0.0])
                )
            edges.append(E)
            for fsi in fs:
                faces[fsi].edgeIndices.append(idx_e)





    return Geometry(
        vertices=vertices,
        normals=normals,
        faces=faces,
        edges=edges,
        adjacency=[]
    )



def loadObjPC(filename,nn=6):
    vertices = []
    normals = []
    faces = []
    edges = []
    adjacency = []
    with open(filename, newline='') as f:
        flines = f.readlines()
        # Read vertices
        indexCounter = 0;
        print('Reading Vertices')
        for row in flines:
            if row[0] == 'v' and row[1] == ' ':
                line = row.rstrip()
                line = line[2:len(line)]
                coords = line.split()
                coords = list(map(float, coords))
                v = Vertex(
                    index=indexCounter,
                    position=np.asarray([coords[0], coords[1], coords[2]]),
                    delta=[],
                    normal=np.asarray([0.0, 0.0, 0.0]),
                    neighbouringFaceIndices=[],
                    neighbouringVerticesIndices=[],
                    theta=0.0,
                    edgeIndices=[],
                    rotationAxis=np.asarray([0.0, 0.0, 0.0]),
                    #color=np.asarray([coords[3], coords[4], coords[5]])
                    color = np.asarray([0.5, 0.5, 0.5])
                )
                indexCounter += 1;
                vertices.append(v)
        V = np.asarray([v.position for v in vertices])
        tree = KDTree(V)
        nearest_dist, nearest_ind = tree.query(V, k=nn + 1)
        for idx_v, v in enumerate(vertices):
            vertices[idx_v]=vertices[idx_v]._replace(neighbouringVerticesIndices=nearest_ind[idx_v,1:])
        # Which faces are neighbouring to each vertex
        # print('Which faces are neighbouring to each vertex')
        # for idx_f, f in enumerate(faces):
        #     for idx_v, v in enumerate(f.verticesIndices):
        #         vertices[v].neighbouringFaceIndices.append(f.index)


    return Geometry(
        vertices=vertices,
        normals=normals,
        faces=faces,
        edges=edges,
        adjacency=[]
    )


def addNoise(Geom, noiseLevel):
    avg_len = 0.0
    for idx_e, e in enumerate(Geom.edges):
        avg_len += e.length
    avg_len = avg_len / len(Geom.edges)
    stddev = avg_len * noiseLevel
    g = np.random.normal(0, stddev, len(Geom.vertices))
    #stddevList=[avg_len * np.random.uniform(low=0.01, high=0.3) for i in range(0,len(Geom.vertices))]
    #muList = [0.0 for i in range(0, len(Geom.vertices))]
    #g = np.random.normal(muList, stddevList, len(Geom.vertices))
    for idx_v, v in enumerate(Geom.vertices):
        Geom.vertices[idx_v].position[0] = Geom.vertices[idx_v].position[0] + Geom.vertices[idx_v].normal[0] * g[idx_v]
        Geom.vertices[idx_v].position[1] = Geom.vertices[idx_v].position[1] + Geom.vertices[idx_v].normal[1] * g[idx_v]
        Geom.vertices[idx_v].position[2] = Geom.vertices[idx_v].position[2] + Geom.vertices[idx_v].normal[2] * g[idx_v]



def exportObj(Geom, filename,color=False):
    F = []
    for f in Geom.faces:
        F.append(np.asarray(f.verticesIndices) + np.ones(np.shape(f.verticesIndices)))
    F = np.asarray(F)

    with open(filename, 'w') as writeFile:
        for v in Geom.vertices:
            line = "v " + str(v.position[0]) + " " + str(v.position[1]) + " " + str(v.position[2])
            if color:
                line=line+" "+str(v.color[0]) + " " + str(v.color[1]) + " " + str(v.color[2])
            writeFile.write(line)
            writeFile.write('\n')


        for j in range(0, np.size(F, axis=0)):
            line = "f " + str(int(F[j, 0])) + " " + str(int(F[j, 1])) + " " + str(int(F[j, 2]))
            writeFile.write(line)
            writeFile.write('\n')
    print('Obj model ' + filename + ' exported')







def exportObjPC(Geom, filename):
    V = []
    F = []
    N = []
    for v in Geom.vertices:
        V.append(v.position)
    for v in Geom.vertices:
        N.append(np.asarray(v.normal))
    for f in Geom.faces:
        F.append(np.asarray(f.verticesIndices) + np.ones(np.shape(f.verticesIndices)))
    V = np.asarray(V)
    N = np.asarray(N)

    with open(filename, 'w') as writeFile:
        for j in range(0, np.size(N, axis=0)):
            line = "vn " + str(N[j, 0]) + " " + str(N[j, 1]) + " " + str(N[j, 2])
            writeFile.write(line)
            writeFile.write('\n')
            line = "v " + str(V[j, 0]) + " " + str(V[j, 1]) + " " + str(V[j, 2])
            writeFile.write(line)
            writeFile.write('\n')


    print('Obj model ' + filename + ' exported')


def exportPLYPC(Geom, filename):
    V = []
    F = []
    N = []
    for v in Geom.vertices:
        V.append(v.position)
    for v in Geom.vertices:
        N.append(np.asarray(v.normal))
    V = np.asarray(V)
    N = np.asarray(N)

    with open(filename, 'w') as writeFile:
        writeFile.write("ply")
        writeFile.write('\n')
        writeFile.write("format ascii 1.0")
        writeFile.write('\n')
        writeFile.write("comment VCGLIB generated")
        writeFile.write('\n')
        writeFile.write("element vertex "+ str(N.shape[0]))
        writeFile.write('\n')
        writeFile.write("property float x")
        writeFile.write('\n')
        writeFile.write("property float y")
        writeFile.write('\n')
        writeFile.write("property float z")
        writeFile.write('\n')
        writeFile.write("property float nx")
        writeFile.write('\n')
        writeFile.write("property float ny")
        writeFile.write('\n')
        writeFile.write("property float nz")
        writeFile.write('\n')
        writeFile.write("element face 0")
        writeFile.write('\n')
        writeFile.write("property list uchar int vertex_indices")
        writeFile.write('\n')
        writeFile.write("end_header")
        writeFile.write('\n')
        for j in range(0, np.size(N, axis=0)):
            line = str(round(V[j, 0])) + " " + str(V[j, 1]) + " " + str(V[j, 2]) + " " + str(N[j, 0]) + " " + str(N[j, 1]) + " " + str(N[j, 2])
            writeFile.write(line)
            writeFile.write('\n')
    print('PLY model ' + filename + ' exported')



def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def cropList(listToProcess,maxElements):
    if len(listToProcess)<(maxElements):
        for i in range(0,maxElements-len(listToProcess)):
            listToProcess.append(listToProcess[0])
    if len(listToProcess) > (maxElements):
        listToProcess=listToProcess[:(maxElements)]
    return listToProcess

def neighboursByVertex(Geom, vertexIndex, numOfNeighbours,useRings=True):
    patchVertices = []
    rings=[]
    patchVertices.append(vertexIndex)
    rings=[[vertexIndex]]
    for i in patchVertices:
        rings.append(np.asarray(Geom.vertices[i].neighbouringVerticesIndices))
        for j in Geom.vertices[i].neighbouringVerticesIndices:
            if len(patchVertices) < numOfNeighbours:
                if j not in patchVertices:
                    patchVertices.append(j)
        # for j in Geom.vertices[i].neighbouringVerticesIndices:
        #     if len(patchVertices) < numOfNeighbours:
        #         patchVertices.append(j)
        #     patchVertices = f7(patchVertices)
    patchVertices = cropList(patchVertices, numOfNeighbours)
    rings=np.asarray(rings)
    rings=np.concatenate(rings).ravel().tolist()
    rings=cropList(rings,numOfNeighbours)
    return patchVertices,rings


def neighboursByFace(Geom, faceIndex, numOfNeighbours,useRings=True):
    #patchFaces = []
    #patchFaces.append(faceIndex)
    rings = [faceIndex]

    # patchFaces = [faceIndex]
    # for i in patchFaces:
    #     for j in Geom.faces[i].neighbouringFaceIndices:
    #        if len(patchFaces) < numOfNeighbours:
    #            if j not in patchFaces:
    #                patchFaces.append(j)


    patchFaces = [faceIndex]
    crosschecklist=set()
    for i in patchFaces:
        for j in Geom.faces[i].neighbouringFaceIndices:
           if len(patchFaces) < numOfNeighbours:
               if j not in crosschecklist:
                   patchFaces.append(j)
                   crosschecklist.add(j)


    # for i in patchFaces:
    #     if len(patchFaces) < numOfNeighbours:
    #         patchFaces.extend(Geom.faces[i].neighbouringFaceIndices)
    #         patchFaces=sorted(set(patchFaces), key=patchFaces.index)
    # ik = 0
    # while ik < numOfNeighbours:
    #     ikk=patchFaces[ik]
    #     patchFaces.extend(Geom.faces[ikk].neighbouringFaceIndices)
    #     patchFaces = sorted(set(patchFaces), key=patchFaces.index)
    #     ik += 1
    # for i in patchFaces:
    #     for j in Geom.faces[i].neighbouringFaceIndices:
    #        if len(patchFaces) < numOfNeighbours:
    #            if j not in patchFaces:
    #                patchFaces.append(j)
    # for i in rings:
    #     if len(rings) <  4*numOfNeighbours:
    #         rings.extend(Geom.faces[i].neighbouringFaceIndices)
    # np.savetxt('test1.csv',patchFaces,delimiter=',')
    # patchFaces=sorted(set(rings), key=rings.index)
    # np.savetxt('test2.csv',patchFaces,delimiter=',')
    # rings=cropList(rings,numOfNeighbours)

    patchFaces=cropList(patchFaces,numOfNeighbours)
    return patchFaces,rings





def updateGeometryConnectivity(Geom):
    print('Updating geometry connectivity')


def updateGeometryAttibutes(Geom, useGuided=False, numOfFacesForGuided=10,computeDeltas=False,computeAdjacency=False,computeVertexNormals=True):
    print('Updating geometry attributes')
    ## Positioning & orientation
    print('Get vertex objects for each face')
    for idx_f, f in enumerate(Geom.faces):
        v = [Geom.vertices[fvi] for fvi in Geom.faces[idx_f].verticesIndices]
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(vertices=v)
    print('Compute centroids')
    for idx_f, f in enumerate(Geom.faces):
        vPos = [Geom.vertices[i].position for i in Geom.faces[idx_f].verticesIndices]
        vPos = np.asarray(vPos)
        centroid = np.mean(vPos, axis=0)
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(centroid=centroid)
    print('Process centroids and normals per face')
    for idx_f, f in enumerate(Geom.faces):
        bc = Geom.faces[idx_f].vertices[1].position - Geom.faces[idx_f].vertices[2].position
        ba = Geom.faces[idx_f].vertices[0].position - Geom.faces[idx_f].vertices[2].position
        normal = np.cross(bc, ba)
        faceArea = 0.5 * np.linalg.norm(normal)
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(area=faceArea)
        if np.linalg.norm(normal)!=0:
            normalizedNormal = normal / np.linalg.norm(normal)
        else:
            normalizedNormal = np.asarray([0.0,1.0,0.0])
            print("Problem")
        Geom.faces[idx_f].faceNormal[0] = normalizedNormal[0]
        Geom.faces[idx_f].faceNormal[1] = normalizedNormal[1]
        Geom.faces[idx_f].faceNormal[2] = normalizedNormal[2]
        if np.linalg.norm(normalizedNormal)==0:
            print('Warning')

    if computeVertexNormals:
        print('Process normals per vertex')
        for idx_v, v in enumerate(Geom.vertices):
            normal = np.asarray([0.0, 0.0, 0.0])
            for idx_f, f in enumerate(Geom.vertices[idx_v].neighbouringFaceIndices):
                normal += Geom.faces[f].faceNormal
            normal = normal / len(Geom.vertices[idx_v].neighbouringFaceIndices)
            normal = normal / np.linalg.norm(normal)
            Geom.vertices[idx_v].normal[0] = normal[0]
            Geom.vertices[idx_v].normal[1] = normal[1]
            Geom.vertices[idx_v].normal[2] = normal[2]

    if useGuided:
        print('Process guided')
        numOfFaces_ = numOfFacesForGuided
        patches = []
        for i in range(0, len(Geom.faces)):
            # print('Start searching patces for face '+str(i)+' '+ str(time.time() - t))
            p = neighboursByFace(Geom, i, numOfFaces_)
            patches.append(p)
            # print('Searching patces for face complete '+str(i)+ ' '+str(time.time() - t))
        for idx_f, f in enumerate(Geom.faces):
            if idx_f != f.index:
                print('Maybe prob', f.index)
            selectedPatches = []
            for p in patches:
                if f.index in p:
                    selectedPatches.append(p)
            patchFactors = []
            for p in selectedPatches:
                patchFaces = [Geom.faces[i] for i in p]
                patchNormals = [pF.faceNormal for pF in patchFaces]
                normalsDiffWithinPatch = [np.linalg.norm(patchNormals[0] - p, 2) for p in patchNormals]
                maxDiff = max(normalsDiffWithinPatch)
                patchNormals = np.asarray(patchNormals)
                M = np.matmul(np.transpose(patchNormals), patchNormals)
                w, v = np.linalg.eig(M)
                eignorm = np.linalg.norm(np.diag(v))
                patchFactor = eignorm * maxDiff
                patchFactors.append(patchFactor)
            minIndex = np.argmin(np.asarray(patchFactors))
            p = selectedPatches[minIndex]
            patchFaces = [Geom.faces[i] for i in p]
            weightedNormalFactors = [pF.area * pF.faceNormal for pF in patchFaces]
            weightedNormalFactors = np.asarray(weightedNormalFactors)
            weightedNormal = np.mean(weightedNormalFactors, axis=0)
            weightedNormal = weightedNormal / np.linalg.norm(weightedNormal)
            Geom.faces[f.index] = Geom.faces[f.index]._replace(guidedNormal=weightedNormal)

    if computeDeltas:
        print("Compute deltas")
        for idx_v, v in enumerate(Geom.vertices):
            neibs=Geom.vertices[idx_v].neighbouringVerticesIndices
            vPos=np.asarray([Geom.vertices[i].position for i in neibs])
            sSum=np.sum(vPos,axis=0)/len(neibs)
            computedDelta=Geom.vertices[idx_v].position-sSum
            Geom.vertices[idx_v]=Geom.vertices[idx_v]._replace(delta=computedDelta)


    # if computeAdjacency:
    #     print("Compute adjacency matrix")
    #     S = dok_matrix((len(Geom.vertices), len(Geom.vertices)), dtype=np.int)
    #     for idx_f, f in enumerate(Geom.faces):
    #         vInds = Geom.faces[idx_f].verticesIndices
    #         S[vInds[0] ,vInds[1]] = 1
    #         S[vInds[1] ,vInds[0]] = 1
    #         S[vInds[1] ,vInds[2]] = 1
    #         S[vInds[2] ,vInds[1]] = 1
    #         S[vInds[0] ,vInds[2]] = 1
    #         S[vInds[2] ,vInds[0]] = 1




def computeAdjacencyMatrix(Geom):
    S = dok_matrix((len(Geom.vertices), len(Geom.vertices)), dtype=np.int)
    for idx_f, f in enumerate(Geom.faces):
        vInds = Geom.faces[idx_f].verticesIndices
        S[vInds[0], vInds[1]] = 1
        S[vInds[1], vInds[0]] = 1
        S[vInds[1], vInds[2]] = 1
        S[vInds[2], vInds[1]] = 1
        S[vInds[0], vInds[2]] = 1
        S[vInds[2], vInds[0]] = 1
    return S


def create_identity_dok_matrix(n,m):
    S = dok_matrix((n, m), dtype=np.int)
    minDimension=min(n,m)
    for i in range(minDimension):
        S[i,i]=1
    return S






def asCartesian(rthetaphi):
    # takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1] * np.pi / 180  # to radian
    phi = rthetaphi[2] * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]


def asSpherical(xyz):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r) * 180 / np.pi  # to degrees
    phi = np.arctan2(y, x) * 180 / np.pi
    if phi < 0:
        phi = phi + 360
    if theta < 0:
        theta = theta + 360
    return [r, theta, phi]


def mat2Sph(M):
    for i in range(0, np.size(M, axis=1)):
        xyz = M[:, i]
        r, theta, phi = asSpherical(xyz)
        M[0, i] = r
        M[1, i] = theta
        M[2, i] = phi
    return M


def mat2Cartesian(M):
    for i in range(0, np.size(M, axis=1)):
        rthetaphi = M[:, i]
        x, y, z = asSpherical(rthetaphi)
        M[0, i] = x
        M[1, i] = y
        M[2, i] = z
    return M


def rotate(a, axis, theta):
    if theta != 0:
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        a = a * costheta + np.cross(axis, a) * sintheta + axis * np.dot(axis, a) * (1 - costheta)
    return a

def rotatePatch(patch,axis,theta):
    if theta != 0:
        # for fIndex in range(np.shape(patch)[0]):
        #    patch[fIndex,]=rotate(patch[fIndex,], axis, theta)
        # patch = np.transpose(patch)

        I = np.eye(3)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)
        for fIndex in range(np.shape(patch)[0]):
            patch[fIndex,] = np.matmul(R, np.transpose(
                patch[fIndex,]))
        patch = np.transpose(patch)


    return patch

def computeRotation(vec, target):
    vec = vec / np.linalg.norm(vec)
    target = target / np.linalg.norm(target)
    theta = math.acos(np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target)))
    axis = np.cross(vec, target)
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    return axis, theta

def computePointCloudNormals(Geom,nNeigb):
    print('Starting normals computation')
    V=np.asarray([v.position for v in Geom.vertices])
    N=np.zeros(np.shape(V))

    tree = KDTree(V)
    nearest_dist, nearest_ind = tree.query(V, k=nNeigb+1)
    nearest_ind=nearest_ind[:,1:]
    for i in range(V.shape[0]):
        if i % 2000 == 0:
            print('Normals computation : ' + str(np.round((100 * i / V.shape[0]), decimals=2)) + ' ' + '%')
        #Geom.vertices[i]=Geom.vertices[i]._replace(neighbouringVerticesIndices=nearest_ind)
        nearest_vertices=V[np.array(nearest_ind[i]),:]
        nearest_vertices=np.transpose(nearest_vertices)
        #d=nearest_vertices-np.matlib.repmat(V[i,:], len(nearest_ind[i]), 1)
        #meanD=np.matlib.repmat(np.mean(d,axis=0), d.shape[0], 1)
        #dm=d-meanD
        #Cov=np.matmul(np.transpose(dm),dm)/nNeigb
        Cov=np.cov(nearest_vertices)
        wi, vi = LA.eig(Cov)
        d=wi
        #d = np.diag(wi)
        index_min = np.argmin(d)
        val_min=d[index_min]
        N[i,:]=np.transpose(vi[:,index_min])
        Geom.vertices[i].normal[0]=  N[i,0]
        Geom.vertices[i].normal[1] = N[i,1]
        Geom.vertices[i].normal[2] = N[i,2]
    adjList=[0]
    for i in adjList:
        for nn in nearest_ind[i, :].tolist():
            if nn not in adjList:
                N[nn] = np.sign(np.dot(N[i, :], N[nn, :])) * N[nn, :]
                adjList.append(nn)
    for i in range(N.shape[0]):
        Geom.vertices[i].normal[0]=  N[i,0]
        Geom.vertices[i].normal[1] = N[i,1]
        Geom.vertices[i].normal[2] = N[i,2]
    print('ok')
    # for nIter in range(60):
    #     Temp = np.zeros(np.shape(N))
    #     for i in range(N.shape[0]):
    #         for nn in range(len(nearest_ind[i,:])):
    #             #Temp[i,:] = Temp[i,:] + np.sign(np.dot(N[i,:],N[nn,:]))*N[nn,:]
    #             #Temp[i, :] =  np.sign(np.dot(N[i, :], N[nn, :])) * N[i, :]
    #             #Temp[i, :] = Temp[i, :] + np.sign(np.dot(N[i,:],N[nearest_ind[i,nn],:]))*N[nearest_ind[i,nn], :]
    #             Temp[i, :] = np.sign(np.dot(N[i, :], N[nearest_ind[i, nn], :])) * N[i, :]
    #     for i in range(N.shape[0]):
    #         Temp[i, :] = Temp[i, :]/np.linalg.norm(Temp[i, :])
    #         N[i, :] = Temp[i, :]
    return V,nearest_ind



def computeThetas(mModel,rModel):
    thetas=np.zeros((len(mModel.faces),1))
    for f_index,f in enumerate(mModel.faces):
        nr=rModel.faces[f_index].faceNormal
        nm=mModel.faces[f_index].faceNormal
        thetas[f_index]=180*(np.arccos(np.dot(nr,nm)/(np.linalg.norm(nr)*np.linalg.norm(nm)))/np.pi)
    return np.mean(thetas)

def computeThetasV2(mModel,rModel):
    thetas=np.zeros((len(mModel.faces),1))
    for f_index,f in enumerate(mModel.faces):
        nr=rModel.faces[f_index].faceNormal
        nm=mModel.faces[f_index].faceNormal
        thetas[f_index]=180*(np.arccos(np.dot(nr,nm)/(np.linalg.norm(nr)*np.linalg.norm(nm)))/np.pi)
    return thetas


def computeNMSE(G,R,axis=0):
    NMSE=0
    if axis==0:
        NMSE = np.linalg.norm(G-R)/np.linalg.norm(G)
        #NMSE = np.linalg.norm(G - R) / np.linalg.norm(G - np.tile(np.mean(G, axis=0), (np.shape(G)[0], 1)))
    return NMSE



def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v






def make_uniform_quantizer(max_val,min_val,nbits):
    print("Start")
    lvl_count = 2**nbits
    step = (max_val - min_val) / (lvl_count - 1)
    def q(val):
        return np.round((np.clip(val, a_min = min_val, a_max = max_val)-min_val)/step)
    def dq(val):
        return np.clip(min_val+(step*np.round(val)),a_min=min_val,a_max=max_val)
    return q,dq



def quantizeArray(A,nbits=12,axis=1):
    mMax=np.max(A)
    mMin=np.min(A)
    q, dq = make_uniform_quantizer(mMax, mMin, nbits)
    for i in range(np.shape(A)[axis]):
        if axis==1:
            A[:,i]=dq(q(A[:,i]))
        if axis==0:
            A[i,:]=dq(q(A[i,:]))
    return A

def computeGL(mModel,i):
    N = mModel.vertices[i].neighbouringVerticesIndices
    sumd = 0
    sumn = np.zeros(np.shape(mModel.vertices[i].position))
    for j in N:
        sumd = sumd + (1 / np.linalg.norm(mModel.vertices[i].position - mModel.vertices[j].position))
        sumn = sumn + (
                    mModel.vertices[j].position / np.linalg.norm(mModel.vertices[i].position - mModel.vertices[j].position))
    GL=mModel.vertices[i].position-sumn/sumd
    return GL


