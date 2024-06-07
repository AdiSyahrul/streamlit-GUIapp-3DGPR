import streamlit as st
import os
import numpy as np
from scipy.io import loadmat, savemat
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time

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
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5, color='crimson')
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
def calculate_psnr(original_data, reconstructed_data, ):
    mse = np.mean((original_data - reconstructed_data) ** 4)
    psnr = 20 * np.log10(1.0/np.sqrt(mse))
    return psnr
def calculate_ssim(original_data, reconstructed_data):
    ssim_value = ssim(original_data, reconstructed_data, data_range=reconstructed_data.max() - reconstructed_data.min(), multichannel=True)
    return ssim_value

def main():
    st.title("3D GPR Visualization")
    uploaded_file = st.file_uploader("Upload a .mat file", type="mat")
    model_files = list_models()
    selected_model = st.selectbox("Select a model for reconstruction", model_files)
    is_noisy = st.checkbox("Uploaded file already has noise")
    if uploaded_file is not None:
        data = loadmat(uploaded_file)['noisy_data']
        normalized_data = normalize_data(data)
        if not is_noisy:
            noise_data = add_vertical_noise(data)
        else:
            noise_data = data
        # noisy_data = add_vertical_noise(normalized_data)
        # noise_data = add_vertical_noise(data)
        model_path = os.path.join('dataset', selected_model)
        model = tf.keras.models.load_model(model_path)

        start_time = time.time()
        reconstructed_data = model.predict(np.expand_dims(noise_data, axis=0))
        prediction_time = time.time() - start_time
        reconstructed_data = np.squeeze(reconstructed_data)
        # 3D Noisy #
        st.markdown("<h2 style='text-align: center;'>3D Noisy</h2>", unsafe_allow_html=True)
        # level_noisy = np.mean(noise_data)
        level_noisy = np.percentile(noise_data, 75)
        # level_noisy = np.percentile(noisy_data, 75)
        verts_noisy, faces_noisy, _, _ = marching_cubes(noise_data, level=level_noisy)
        mesh_noisy = create_3d_mesh_plot(verts_noisy, faces_noisy)
        st.plotly_chart(go.Figure(data=[mesh_noisy]), use_container_width=True)

        # 3D Reconstructed #
        st.markdown("<h2 style='text-align: center;'>3D Reconstructed</h2>", unsafe_allow_html=True)
        level_reconstructed = np.percentile(reconstructed_data, 75)
        verts_rec, faces_rec, _, _ = marching_cubes(reconstructed_data, level=level_reconstructed)
        mesh_rec = create_3d_mesh_plot(verts_rec, faces_rec)
        st.plotly_chart(go.Figure(data=[mesh_rec]), use_container_width=True)
        
        rec = normalize_data(reconstructed_data)
        # st.write(f"Prediction time: {prediction_time:.2f} seconds")
        if not is_noisy:
            psnr_value = calculate_psnr(normalized_data, rec)
            ssim_value = calculate_ssim(normalized_data, rec)
            st.write(f"PSNR Reconstructed: {psnr_value:.2f}")
            st.write(f"SSIM Reconstructed: {ssim_value:.2f}")
        # st.subheader("2D Visualization")
        # axis = st.selectbox("Select axis for 2D slice visualization", ['x', 'y', 'z'])
        # if axis == 'z':
        #     mip_original = np.max(normalized_data, axis=0)
        #     mip_noisy = np.max(normalize_data(noise_data), axis=0)
        #     mip_reconstructed = np.max(rec, axis=0)
        # elif axis == 'y':
        #     mip_original = np.max(normalized_data, axis=1)
        #     mip_noisy = np.max(normalize_data(noise_data), axis=1)
        #     mip_reconstructed = np.max(rec, axis=1)
        # elif axis == 'x':
        #     mip_original = np.max(normalized_data, axis=2)
        #     mip_noisy = np.max(normalize_data(noise_data), axis=2)
        #     mip_reconstructed = np.max(rec, axis=2)
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.image(mip_original, use_column_width=True, caption="Original Data")
        # with col2:
        #     st.image(mip_noisy, use_column_width=True, caption="Noisy Data")
        # with col3:
        #     st.image(mip_reconstructed, use_column_width=True, caption="Reconstructed Data")
if __name__ == "__main__":
    main()
