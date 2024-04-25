import streamlit as st
import os
import numpy as np
from scipy.io import loadmat
from skimage.measure import marching_cubes
import plotly.graph_objects as go

def load_file(directory, filename):
    file_path = os.path.join(directory, filename)
    data = loadmat(file_path)['noisy_data']
    return data

def create_3d_mesh_plot(data):
    original_image_squeezed = np.squeeze(data)
    level_original = np.mean(original_image_squeezed)
    verts_original, faces_original, _, _ = marching_cubes(original_image_squeezed, level=level_original)

    x, y, z = verts_original.T
    i, j, k = faces_original.T

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5, color='lightpink')
    fig = go.Figure(data=[mesh])
    return fig

def create_slice_plot(data, axis, slice_idx):
    if axis == 'x':
        slice_data = data[slice_idx, :, :]
    elif axis == 'y':
        slice_data = data[:, slice_idx, :]
    elif axis == 'z':
        slice_data = data[:, :, slice_idx]
    slice_data = np.flipud(slice_data)
    fig = go.Figure(data=go.Heatmap(z=slice_data, colorscale='gray'))
    return fig

# Main function
def main():
    st.title("3D Mesh and Slice Visualization")

    # Get the uploaded file
    uploaded_file = st.file_uploader("Upload a .mat file", type="mat")

    # Load and display the 3D mesh if file is uploaded
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        single_data = loadmat(uploaded_file)['noisy_data']

        # Display the 3D mesh visualization
        st.subheader("3D Mesh Visualization")
        mesh_plot = create_3d_mesh_plot(single_data)
        st.plotly_chart(mesh_plot)

        # Get user input for slice axis and index
        st.subheader("2D Slice Visualization")
        axis = st.selectbox("Select slice axis", ['x', 'y', 'z'])
        slice_idx = st.slider(f"Select slice index along {axis} axis", 0, single_data.shape[0]-1, single_data.shape[0]//2)

        # Create and display the slice plot for the selected slice
        slice_plot = create_slice_plot(single_data, axis, slice_idx)
        st.plotly_chart(slice_plot)

        # st.write(f"Slice index: {slice_idx}")
        # st.write(f"Slice axis: {axis}")

if __name__ == "__main__":
    main()

