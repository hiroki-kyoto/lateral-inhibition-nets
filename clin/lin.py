import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pickle
plt.style.use('ggplot')
def getCIFAR10(i):
    return unpickle('./cifar-10-batches-py/data_batch_'+i)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def runBinarization(x, r, ratio):
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s = 0
            t = 0
            for k in range(max(0,i-r),min(x.shape[0],i+r)):
                for h in range(max(0,j-r),min(x.shape[1],j+r)):
                    t += 1.0
                    if x[i][j][0] > x[k][h][0]:
                        s += 1.0
            if s/t >= ratio:
                y[i][j][:] = 1.0
            else:
                y[i][j][:] = 0.0
    return y
def runLateralInhibition(x, r): # r is radius for a neighbor definition
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s = 0
            for k in range(max(0,i-r),min(i+r,x.shape[0])):
                for h in range(max(0,j-r),min(j+r,x.shape[1])):
                    s += x[k][h][0]
            if s > 0:
                y[i][j][:] = x[i][j][0]/max(1.0,s)
            else:
                y[i][j][:] = 0
    return y
################# main procedure begin ###################
def createInceptionFilters(count, depth):
    return np.random.randn(count, depth)
def CreateConvolutionFilters(count, height, width):
    return np.random.randn(count, height, width)
def applyInceptionFilters(filters, feature_maps):
    if filters.shape[1] != feature_maps.shape[0]:
        raise NameError('Inception filter dimension is invalid!')
    d = filters.shape[0]
    h = feature_maps.shape[1]
    w = feature_maps.shape[2]
    # prepare multiplication operation
    m = feature_maps.transpose([1,2,0])
    new_maps = np.zeros([d, h, w])
    for i in range(filters.shape[0]):
        new_maps[i,:,:] = m.dot(filters[i])
    return new_maps
def conv(m, f):
    h = m.shape[0]-f.shape[0] + 1
    w = m.shape[1]-f.shape[1] + 1
    nm = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            nm[i,j] = np.sum(m[i:i+f.shape[0],j:j+f.shape[1]]*f)
    return nm
def applyConvolutionFilters(filters, feature_maps):
    nm = feature_maps.shape[0]
    nf = filters.shape[0]
    d = nm * nf
    h = feature_maps.shape[1] - filters.shape[1] + 1
    w = feature_maps.shape[2] - filters.shape[2] + 1
    new_maps = np.zeros([d,h,w])
    for i in range(nm):
        for j in range(nf):
            new_maps[i*nf+j,:,:] = conv(feature_maps[i], filters[j])
    return new_maps
def applyLateralInhibition(feature_maps, radius):
    return 0
    #for i in range(feature_maps.shape[0]):
#################### TEST O CIFAR-10 ################
cifar_10_data = getCIFAR10('1')
im = np.zeros([8,32,32,3])
for iid in range(8):
    for i in range(32):
        for k in range(32):
            im[iid][i][k][0] = cifar_10_data[b'data'][iid][(i*32+k)]/255.0
            im[iid][i][k][1] = cifar_10_data[b'data'][iid][1024+(i*32+k)]/255.0
            im[iid][i][k][2] = cifar_10_data[b'data'][iid][2048+(i*32+k)]/255.0
# show original images
f, axs = plt.subplots(1, 8, sharey=True,figsize=(8,1))
for iid in range(8):
    axs[iid].axis('off')
    axs[iid].imshow(im[iid])
f.show(1)
x = np.random.randn(3,3)
f = np.array([[1,0],[0,1]])
y = conv(x, f)
print(y)
