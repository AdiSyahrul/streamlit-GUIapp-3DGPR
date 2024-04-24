# pyglet_renderer.py
import pyglet
import numpy as np
import trimesh
from scipy.io import loadmat
from skimage.measure import marching_cubes
import sys

def load_file(file_path):
    data = loadmat(file_path)['noisy_data']
    return data

def render_mesh(data_file):
    data = load_file(data_file)
    original_image_squeezed = np.squeeze(data)
    level_original = np.mean(original_image_squeezed)
    verts_original, faces_original, _, _ = marching_cubes(original_image_squeezed, level=level_original)
    mesh_original = trimesh.Trimesh(vertices=verts_original, faces=faces_original)

    window = pyglet.window.Window()
    
    @window.event
    def on_draw():
        window.clear()
        mesh_original.show()
    
    pyglet.app.run()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pyglet_renderer.py <data_file>")
    else:
        data_file = sys.argv[1]
        render_mesh(data_file)
