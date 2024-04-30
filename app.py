import streamlit as st
import os
import numpy as np
from scipy.io import loadmat
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import tensorflow as tf
import matplotlib.pyplot as plt

def add_vertical_noise(data, noise_ratio=0.3, line_width=3):
    noisy_data = data.copy()
    x_dim, y_dim, z_dim = noisy_data.shape
    num_noisy_lines = int((y_dim * noise_ratio) / line_width)
    noise_color = np.min(noisy_data)
    # noise_color = np.percentile(noisy_data[noisy_data > 0], 25)
    start_cols = np.random.choice(y_dim - line_width + 1, num_noisy_lines, replace=False)

    for start_col in start_cols:
        end_col = start_col + line_width
        end_col = end_col if end_col <= y_dim else y_dim
        noisy_data[:, start_col:end_col, :] = noise_color

    return noisy_data

def load_file(directory, filename):
    file_path = os.path.join(directory, filename)
    data = loadmat(file_path)['noisy_data']
    return data

# def create_3d_mesh_plot(data):
#     original_image_squeezed = np.squeeze(data)
#     level_original = np.mean(original_image_squeezed)
#     # level_original = np.mean(data)
#     verts_original, faces_original, _, _ = marching_cubes(original_image_squeezed, level=level_original)

#     x, y, z = verts_original.T
#     i, j, k = faces_original.T

#     mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5, color='gray')
#     fig = go.Figure(data=[mesh])
#     return fig
def create_3d_mesh_plot(points, faces):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5, color='lightpink')
    return mesh

def create_slice_plots(data, axis, start_slice, num_slices):
    slice_plots = []
    for i in range(start_slice, start_slice + num_slices):
        if axis == 'x':
            slice_data = data[i, :, :]
        elif axis == 'y':
            slice_data = data[:, i, :]
        elif axis == 'z':
            slice_data = data[:, :, i]
        slice_data = np.flipud(slice_data)
        fig = go.Figure(data=go.Heatmap(z=slice_data, colorscale='gray'))
        slice_plots.append(fig)
    return slice_plots

# def reconstruct_data(noisy_data, model):
#     # reconstructed_data = model.predict(np.expand_dims(noisy_data, axis=0))[0]
#     reconstructed_data = model.predict(noisy_data)
#     return reconstructed_data
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
# Main function
def main():
    st.title("3D Mesh and Slice Visualization")
    uploaded_file = st.file_uploader("Upload a .mat file", type="mat")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        single_data = loadmat(uploaded_file)['noisy_data']
        normalized_data = normalize_data(single_data)
        # Add noise to the uploaded data
        noisy_data = add_vertical_noise(normalized_data)
        noise_data = add_vertical_noise(single_data)

        # 3D Noisy
        st.subheader("3D Mesh Noisy Visualization")
        # level = np.mean(noisy_data)
        verts, faces, _, _ = marching_cubes(noise_data)
        mesh = create_3d_mesh_plot(verts, faces)
        fig = go.Figure(data=[mesh])
        st.plotly_chart(fig)

        # 3D Reconstruction
        model = tf.keras.models.load_model('dataset/model3tes2.h5')
        reconstructed_data = model.predict(np.expand_dims(noisy_data, axis=0))[0]
        reconstructed_data = np.squeeze(reconstructed_data)
        print("New shape of reconstructed data:", reconstructed_data.shape)

        st.subheader("Reconstructed Data")
        verts_reconstructed, faces_reconstructed, _, _ = marching_cubes(reconstructed_data)
        mesh_reconstructed = create_3d_mesh_plot(verts_reconstructed, faces_reconstructed)
        fig_reconstructed = go.Figure(data=[mesh_reconstructed])
        st.plotly_chart(fig_reconstructed)

        # Maximum Intensity Projection (MIP) of the original data
        mip_original = np.max(normalized_data, axis=2)
        mip_noisy = np.max(noisy_data, axis=2)
        mip_reconstructed = np.max(reconstructed_data, axis=2)
        # mip_original_normalized = (mip_original - np.min(mip_original)) / (np.max(mip_original) - np.min(mip_original))
        # mip_noisy_normalized = (mip_noisy - np.min(mip_noisy)) / (np.max(mip_noisy) - np.min(mip_noisy))
        # mip_reconstructed_normalized = (mip_reconstructed - np.min(mip_reconstructed)) / (np.max(mip_reconstructed) - np.min(mip_reconstructed))

        #  Display MIPs
        st.subheader("Original Data MIP")
        st.image(mip_original, use_column_width=True, caption="Maximum Intensity Projection of the Original Data")

        st.subheader("Noisy Data MIP")
        st.image(mip_noisy, use_column_width=True, caption="Maximum Intensity Projection of the Noisy Data")

        st.subheader("Reconstructed Data MIP")
        st.image(mip_reconstructed, use_column_width=True, caption="Maximum Intensity Projection of the Reconstructed Data")

if __name__ == "__main__":
    main()

