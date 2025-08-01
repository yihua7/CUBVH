import os
import zlib
import glob
import tqdm
import time
import mcubes
import trimesh
import argparse
import numpy as np

import torch
import cubvh

import kiui

"""
Extract watertight mesh from a arbitrary mesh by UDF expansion and floodfill.
"""
parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('--res', type=int, default=512)
parser.add_argument('--workspace', type=str, default='output')
parser.add_argument('--spmc', action='store_true') # use sparse CUDA marching cubes, usually faster.
parser.add_argument('--target_faces', type=int, default=1000000)
opt = parser.parse_args()

device = torch.device('cuda')

points = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, opt.res, device=device),
        torch.linspace(-1, 1, opt.res, device=device),
        torch.linspace(-1, 1, opt.res, device=device),
        indexing="ij",
    ), dim=-1,
) # [N, N, N, 3]


def sphere_normalize(vertices):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    radius = np.linalg.norm(vertices - bcenter, axis=-1).max()
    vertices = (vertices - bcenter) / (radius)  # to [-1, 1]
    return vertices


def box_normalize(vertices, bound=0.95):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices


def run(path):

    mesh = trimesh.load(path, process=False, force='mesh')
    mesh.vertices = box_normalize(mesh.vertices, bound=0.95)
    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    triangles = torch.from_numpy(mesh.faces).long().to(device)

    t0 = time.time()
    BVH = cubvh.cuBVH(vertices, triangles)
    print(f'BVH build time: {time.time() - t0:.4f}s')
    eps = 2 / opt.res

    ### udf
    t0 = time.time()
    udf, _, _ = BVH.unsigned_distance(points.view(-1, 3), return_uvw=False)
    print(f'UDF time: {time.time() - t0:.4f}s')

    ### floodfill
    t0 = time.time()
    udf = udf.view(opt.res, opt.res, opt.res).contiguous()
    occ = udf < eps
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    print(f'Floodfill time: {time.time() - t0:.4f}s')

    ### binary occupancy
    occ_mask = ~empty_mask

    ### truncated SDF
    sdf = udf - eps  # inner is negative
    inner_mask = occ_mask & (sdf > 0)
    sdf[inner_mask] *= -1

    print(f'SDF occupancy ratio: {torch.sum(sdf < 0) / opt.res ** 3:.4f}')

    ### sparse marching cubes
    if opt.spmc:
        from cubvh import sparse_marching_cubes
        # convert to sparse voxels (not counting time for this part)
        sdf_000 = sdf[:-1, :-1, :-1]
        sdf_001 = sdf[:-1, :-1, 1:]
        sdf_010 = sdf[:-1, 1:, :-1]
        sdf_011 = sdf[:-1, 1:, 1:]
        sdf_100 = sdf[1:, :-1, :-1]
        sdf_101 = sdf[1:, :-1, 1:]
        sdf_110 = sdf[1:, 1:, :-1]
        sdf_111 = sdf[1:, 1:, 1:]
        # only keep voxels where the 8 corners have different sign
        sdf_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
        sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
        sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
        sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
        sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
        sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
        sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)
        
        active_cells_index = torch.nonzero(sdf_mask, as_tuple=True) # ([N], [N], [N])
        active_cells = torch.stack(active_cells_index, dim=-1) # [N, 3]
        
        active_cells_sdf = torch.stack([
            sdf_000[active_cells_index],
            sdf_100[active_cells_index],
            sdf_110[active_cells_index],
            sdf_010[active_cells_index],
            sdf_001[active_cells_index],
            sdf_101[active_cells_index],
            sdf_111[active_cells_index],
            sdf_011[active_cells_index],
        ], dim=-1) # [N, 8]
        
        t0 = time.time()
        vertices, triangles = sparse_marching_cubes(active_cells, active_cells_sdf, 0)
        vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1
        watertight_mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), triangles.detach().cpu().numpy())
        print(f'Sparse MC time: {time.time() - t0:.4f}s, vertices: {len(watertight_mesh.vertices)}, triangles: {len(watertight_mesh.faces)}')
    else:
        ### dense marching cubes
        t0 = time.time()
        vertices, triangles = mcubes.marching_cubes(sdf.cpu().numpy(), 0)
        vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        watertight_mesh = trimesh.Trimesh(vertices, triangles)
        print(f'MC time: {time.time() - t0:.4f}s, vertices: {len(watertight_mesh.vertices)}, triangles: {len(watertight_mesh.faces)}')
    
    ### decimation
    if opt.target_faces > 0:
        t0 = time.time()
        from kiui.mesh_utils import decimate_mesh
        vertices, faces = decimate_mesh(watertight_mesh.vertices, watertight_mesh.faces, opt.target_faces)
        watertight_mesh = trimesh.Trimesh(vertices, faces)
        print(f'Decimation time: {time.time() - t0:.4f}s, vertices: {len(watertight_mesh.vertices)}, triangles: {len(watertight_mesh.faces)}')

    ### output
    name = os.path.splitext(os.path.basename(path))[0]
    watertight_mesh.export(f'{opt.workspace}/{name}.obj')

os.makedirs(opt.workspace, exist_ok=True)

if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    for path in tqdm.tqdm(file_paths):
        try:
            run(path)
        except Exception as e:
            print(f'[WARN] {path} failed: {e}')
else:
    run(opt.test_path)