import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

class_name = "Pistol"
no_seg = False

import os
root = os.path.abspath(".") + "/saved_results/"

def plot_ptcloud(ptcloud, seg, title):
    fig = plt.figure()
    fig.suptitle(class_name + " " + title)
    ax = fig.add_subplot(111, projection='3d')
    plt.axis('off')

    if no_seg:
        pts = ptcloud
        xs = pts[:,0]
        ys = pts[:,1]
        zs = pts[:,2]
        ax.scatter(xs, ys, zs)
    else:
        for i in range(6):
            pts = (ptcloud[seg == i])
            xs = pts[:,0]
            ys = pts[:,1]
            zs = pts[:,2]
            ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title(title)
    fig.show()

orig = np.load(root + "original_" + class_name + ".npy")
orig_seg = np.load(root + "original_seg_" + class_name + ".npy")
crop = np.load(root + "cropped_" + class_name + ".npy")
sparse = np.load(root + "sparse_" + class_name + ".npy")
sparse_seg = np.load(root + "sparse_seg_" + class_name + ".npy")
dense = np.load(root + "dense_" + class_name + ".npy")
dense_seg = np.load(root + "dense_seg_" + class_name + ".npy")

bs = orig.shape[0]

for i in range(bs):
    plot_ptcloud(orig[i], orig_seg[i], "original")
    plot_ptcloud(crop[i], orig_seg[i], "cropped")
    plot_ptcloud(sparse[i], np.argmax(sparse_seg[i], axis=1), "sparse")
    plot_ptcloud(dense[i], np.argmax(dense_seg[i], axis=1), "dense")
    plt.show()
    txt = input("do you want to cont? (y/n)")
    if txt != "y":
        break
    
