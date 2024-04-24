import streamlit as st
import os
import numpy as np
import trimesh
from scipy.io import loadmat
from skimage.measure import marching_cubes

def load_file(directory, filename):
    file_path = os.path.join(directory, filename)
    data = loadmat(file_path)['noisy_data']
    return data

def main():
    st.title("3D Mesh Visualization")

    uploaded_file = st.file_uploader("Upload a .mat file", type="mat")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        bytes_data = uploaded_file.read()
        single_data = loadmat(uploaded_file)['noisy_data']

        original_image_squeezed = np.squeeze(single_data)
        level_original = np.mean(original_image_squeezed)
        verts_original, faces_original, _, _ = marching_cubes(original_image_squeezed, level=level_original)
        mesh_original = trimesh.Trimesh(vertices=verts_original, faces=faces_original)

        st.write("Original Data")
        st.write(mesh_original.show())
    
if __name__ == "__main__":
    main()
