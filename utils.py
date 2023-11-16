import numpy as np
import pytorch3d
from pytorch3d.structures import Meshes
import trimesh
from trimesh.exchange.obj import export_obj
from scipy.spatial import cKDTree
import torch


import torch
from pytorch3d.structures import Meshes

def getSubMesh(v, f, vertex_indices):
    # Ensure v and f are lists of tensors
    mesh = Meshes(verts=v, faces=f)

    # Determine the device from the input tensor
    device = v[0].device

    # Convert vertex_indices to a set for faster lookup
    vertex_indices_set = set(vertex_indices)

    # Get the faces of the original mesh
    faces = mesh.faces_list()[0]

    # Find faces where all three vertices are in the subset
    mask = [all(vertex.item() in vertex_indices_set for vertex in face) for face in faces]
    submesh_faces = faces[mask]

    # Create a mapping from old vertex indices to new ones
    new_vertex_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}

    # Re-map vertex indices for the new faces and ensure it's on the same device
    remapped_faces = torch.tensor([[new_vertex_indices[vertex.item()] for vertex in face] for face in submesh_faces], device=device)

    # Extract the vertex positions for the submesh
    submesh_verts = v[0][vertex_indices]

    # Adjust dimensions to match the original mesh's structure
    sub_verts = submesh_verts.unsqueeze(0)  # shape: [1, num_verts, 3]
    sub_faces = remapped_faces.unsqueeze(0)  # shape: [1, num_faces, 3]

    return sub_verts, sub_faces

    # Example usage:
    # v and f are the vertices and faces of your original mesh as lists of tensors.
    # vertex_indices is a 1D tensor or list of indices for the vertices you want to include in the submesh.

def compute_normal(v, f):
    """v, f: Tensors"""
    
    normal = Meshes(verts=list(v),
                    faces=list(f)).verts_normals_list()[0]
    return normal.unsqueeze(0)


def save_mesh_obj(v, f, n, path):
    mesh_save = trimesh.Trimesh(vertices=v,
                                faces=f,
                                vertex_normals=n)
    obj_save  = export_obj(mesh_save, include_normals=True)
    with open(path, "w") as file:
        print(obj_save, file=file)


def compute_distance(v_pred, v_gt, f_pred, f_gt, n_samples=150000):
    """
    The results are evaluated based on three distances:
    1. Chamfer Distance (CD)
    2. Average Absolute Distance (AD)
    3. Hausdorff Distance (HD)

    Please see DeepCSR paper in details:
    https://arxiv.org/abs/2010.11423
    
    For original code, please see:
    https://bitbucket.csiro.au/projects/CRCPMAX/repos/deepcsr/browse/eval.py
    """
    
    # chamfer distance
    cd = 0
    kdtree = cKDTree(v_pred)
    cd += kdtree.query(v_gt)[0].mean()/2
    kdtree = cKDTree(v_gt)
    cd += kdtree.query(v_pred)[0].mean()/2

    # AD & HD
    mesh_pred = trimesh.Trimesh(vertices=v_pred, faces=f_pred)
    pts_pred = mesh_pred.sample(n_samples)
    mesh_gt = trimesh.Trimesh(vertices=v_gt, faces=f_gt)
    pts_gt = mesh_gt.sample(n_samples)

    _, P2G_dist, _ = trimesh.proximity.closest_point(mesh_pred, pts_gt)
    _, G2P_dist, _ = trimesh.proximity.closest_point(mesh_gt, pts_pred)

    # average absolute distance
    assd = ((P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size))
    
    # Hausdorff distance
    hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))

    return cd, assd, hd
