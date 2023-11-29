import pyvista as pv
import os
import numpy as np
from scipy.spatial import cKDTree

def vertex_hausdorff_distance(mesh1, mesh2):
    # Compute per-vertex Hausdorff distance
    tree1 = cKDTree(mesh1.points)
    tree2 = cKDTree(mesh2.points)
    dist1, _ = tree1.query(mesh2.points)
    dist2, _ = tree2.query(mesh1.points)
    return dist2  # distances for each vertex in mesh1 to nearest in mesh2

def rotate_mesh(mesh, angle, axis='x'):
    # Rotate the mesh about a specified axis by a given angle (in degrees)
    if axis == 'x':
        mesh.rotate_x(angle)
    elif axis == 'y':
        mesh.rotate_y(angle)
    elif axis == 'z':
        mesh.rotate_z(angle)
    return mesh

def display_models(subject_id, files_directory):
    model_files = {}
    ground_truth_files = {}
    all_distances = []

    # Iterate through files in the directory
    for file in os.listdir(files_directory):
        if file.endswith(".obj") and subject_id in file:
            parts = file.split('_')
            model_name, hemisphere, layer = parts[0], parts[2], int(parts[1].replace('layer', ''))
            key = (model_name, hemisphere, layer)

            if 'gt' in file:
                ground_truth_files[key] = file
            else:
                if key not in model_files:
                    model_files[key] = []
                model_files[key].append(file)

    # Compute distances and find global third quartile
    for (model_name, hemisphere, layer), filenames in sorted(model_files.items(), key=lambda x: x[0][2]):
        for filename in filenames:
            prediction_mesh = pv.read(os.path.join(files_directory, filename))
            if (model_name, hemisphere, layer) in ground_truth_files:
                gt_filename = ground_truth_files[(model_name, hemisphere, layer)]
                gt_mesh = pv.read(os.path.join(files_directory, gt_filename))
                dist = vertex_hausdorff_distance(prediction_mesh, gt_mesh)
                all_distances.extend(dist)

    global_third_quartile_dist = np.percentile(all_distances, 75)


    num_subplots = len(model_files)
    # Set up the plotter with a specified window size
    # Adjust these values as needed to fit your screen and reduce whitespace
    window_width = 300 * num_subplots  # Example width - change as needed
    window_height = 800  # Example height - change as needed
    plotter = pv.Plotter(shape=(1, num_subplots), window_size=(window_width, window_height))
    
    plotter.background_color = 'white'
    column = 0

    for (model_name, hemisphere, layer), filenames in sorted(model_files.items(), key=lambda x: x[0][2]):
        for filename in filenames:
            prediction_mesh = pv.read(os.path.join(files_directory, filename))
            if (model_name, hemisphere, layer) in ground_truth_files:
                gt_filename = ground_truth_files[(model_name, hemisphere, layer)]
                gt_mesh = pv.read(os.path.join(files_directory, gt_filename))

                # Use previously calculated distances
                dist = vertex_hausdorff_distance(prediction_mesh, gt_mesh)
                prediction_mesh['VertexHausdorffDistance'] = dist

                rotated_mesh = rotate_mesh(prediction_mesh, 90, axis='y')
                rotated_mesh = rotate_mesh(rotated_mesh, 90, axis='z')


                plotter.subplot(0, column)
                plotter.add_mesh(rotated_mesh, scalars='VertexHausdorffDistance', cmap='inferno', clim=(0, global_third_quartile_dist))
            else:
                rotated_mesh = rotate_mesh(prediction_mesh, 180, axis='x')
                plotter.subplot(0, column)
                plotter.add_mesh(rotated_mesh, color='indigo')

            plotter.add_text(f"{model_name}", position='upper_edge', color='black', font_size=14)
            plotter.add_text(f"GNN Layers {layer}", position='lower_edge', color='black', font_size=14)
            # plotter.scalar_bars.clear()

            # camera_position = rotated_mesh.center + np.array([0, 0, 1])
            # plotter.camera.position = camera_position
            # plotter.camera.focal_point = rotated_mesh.center
            # plotter.camera.view_up = [0, 1, 0]
            # plotter.reset_camera()


            column += 1

    # Add a single scalar bar for the entire plot
    # plotter.scalar_bars.clear()
    plotter.add_scalar_bar(title="Hausdorff Dist", label_font_size=16, title_font_size=16, n_labels=3, shadow=True, vertical=False)
    plotter.show(auto_close=False)
    plotter.screenshot('brains.png')
    plotter.close()



# Example Usage
files_directory = '/home/william/websurf/test/'
subject_id = '100206'
display_models(subject_id, files_directory)
