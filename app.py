import streamlit as st
import os
import numpy as np
from scipy.io import loadmat, savemat
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import tensorflow as tf
import matplotlib.pyplot as plt

def list_models(directory='dataset'):
    """List all model files in the specified directory"""
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    return model_files

def add_vertical_noise(data, noise_ratio=0.3, line_width=3):
    noisy_data = data.copy()
    x_dim, y_dim, z_dim = noisy_data.shape
    num_noisy_lines = int((y_dim * noise_ratio) / line_width)
    noise_color = np.median(noisy_data[noisy_data < 0])
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
            axis_label = 'X-axis'
        elif axis == 'y':
            slice_data = data[:, i, :]
            axis_label = 'Y-axis'
        elif axis == 'z':
            slice_data = data[:, :, i]
            axis_label = 'Z-axis'
        slice_data = np.flipud(slice_data)
        fig = go.Figure(data=go.Heatmap(z=slice_data, colorscale='gray'))
        slice_plots.append((fig, axis_label))
    return slice_plots

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_file(data, base_filename):
    """Save reconstructed data with a new filename based on the uploaded file name."""
    output_filename = 'reconstructed_' + base_filename + '.mat'
    savemat(output_filename, {'reconstructed_data': data})
    return output_filename

def main():
    st.title("3D GPR Visualization")
    uploaded_file = st.file_uploader("Upload a .mat file", type="mat")
    model_files = list_models()
    selected_model = st.selectbox("Select a model for reconstruction", model_files)
    if uploaded_file is not None:
        base_filename = os.path.splitext(uploaded_file.name)[0]
        st.write(f"File uploaded successfully: {uploaded_file.name}")
        single_data = loadmat(uploaded_file)['noisy_data']
        contains_noise = np.max(single_data) > 1.0

        if not contains_noise:
                # 3D Noisy Visualization
                st.subheader("3D Mesh Noisy Visualization")
                noise_data = add_vertical_noise(single_data)
                verts, faces, _, _ = marching_cubes(noise_data)
                mesh = create_3d_mesh_plot(verts, faces)
                fig = go.Figure(data=[mesh])
                st.plotly_chart(fig)
        else:
            # 3D Noise Visualization
            st.subheader("3D Mesh Noisy Visualization")
            verts, faces, _, _ = marching_cubes(single_data)
            mesh = create_3d_mesh_plot(verts, faces)
            fig = go.Figure(data=[mesh])
            st.plotly_chart(fig)

        # 3D Reconstruction
        model_path = os.path.join('dataset', selected_model)
        model = tf.keras.models.load_model(model_path)
        # model = tf.keras.models.load_model('dataset/model3tes3.h5')
        try:
            reconstructed_data = model.predict(np.expand_dims(noise_data, axis=0))[0]
        except Exception as e:
            st.error(f"Error during model prediction: {str(e)}")
            return
        reconstructed_data = np.squeeze(reconstructed_data)
        print("New shape of reconstructed data:", reconstructed_data.shape)
        st.subheader("Reconstructed Data")
        verts_reconstructed, faces_reconstructed, _, _ = marching_cubes(reconstructed_data)
        mesh_reconstructed = create_3d_mesh_plot(verts_reconstructed, faces_reconstructed)
        fig_reconstructed = go.Figure(data=[mesh_reconstructed])
        st.plotly_chart(fig_reconstructed)
        if st.button('Save Reconstructed Data'):
            save_path = save_file(reconstructed_data, base_filename)
            st.success(f'Reconstructed data saved successfully at {save_path}!')

        # st.subheader("2D Visualization")
        # axis = st.selectbox("Select axis for 2D slice visualization", ['x', 'y', 'z'])
        # if axis == 'z':
        #     mip_original = np.max(normalize_data(single_data), axis=0)
        #     mip_noisy = np.max(normalize_data(noise_data), axis=0)
        #     mip_reconstructed = np.max(reconstructed_data, axis=0)
        # elif axis == 'y':
        #     mip_original = np.max(normalize_data(single_data), axis=1)
        #     mip_noisy = np.max(normalize_data(noise_data), axis=1)
        #     mip_reconstructed = np.max(reconstructed_data, axis=1)
        # elif axis == 'x':
        #     mip_original = np.max(normalize_data(single_data), axis=2)
        #     mip_noisy = np.max(normalize_data(noise_data), axis=2)
        #     mip_reconstructed = np.max(reconstructed_data, axis=2)
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.image(mip_original, use_column_width=True, caption="Original Data")
        # with col2:
        #     st.image(mip_noisy, use_column_width=True, caption="Noisy Data")
        # with col3:
        #     st.image(mip_reconstructed, use_column_width=True, caption="Reconstructed Data")

if __name__ == "__main__":
    main()

