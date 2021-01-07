import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

class_name = "Pistol"

def plot_ptcloud(ptcloud, title):
    fig = plt.figure()
    fig.suptitle(class_name + " " + title)
    ax = fig.add_subplot(111, projection='3d')
    plt.axis('off')

    pts = ptcloud
    xs = pts[:,0]
    ys = pts[:,1]
    zs = pts[:,2]
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title(title)
    fig.show()

orig = np.load(class_name + "_orig.npy")
cropped = np.load(class_name + "_cropped.npy")

plot_ptcloud(orig, "complete")
plot_ptcloud(cropped, "missing")
plt.show()