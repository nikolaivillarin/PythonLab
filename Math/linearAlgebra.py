from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

u = np.array([2, 5])
v = np.array([3, 1])

# zip function returns a tuple of x and y coordinates.
# from the example x_coords = (2, 3) and y_coords = (5, 1)
# zip can handle more than 2 values in an array. To handle more such as 3 values
# per array just assign more tuple values. E.G: a, b, c = zip(u, v)
x_coords, y_coords = zip(u, v)

plt.scatter(x_coords, y_coords, color=["r", "b"])
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
        head_width=0.2, head_length=0.3, length_includes_head=True, **options)

plot_vector2d(u, color="r")
plot_vector2d(v, color="b")
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

a = np.array([1, 2, 8])
b = np.array([5, 6, 3])

subplot3d = plt.subplot(111, projection='3d')

x_coords, y_coords, z_coords = zip(a, b)

subplot3d.scatter(x_coords, y_coords, z_coords)
subplot3d.set_zlim3d([0, 9])

plt.show()