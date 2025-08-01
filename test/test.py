import os
import numpy as np
import trimesh
import torch

import cubvh

cubvh_path = './bvh.pt'

if not os.path.exists(cubvh_path):
    ### build BVH from mesh
    mesh = trimesh.load('mesh2.ply', process=False)
    # NOTE: you need to normalize the mesh first, since the max distance is hard-coded to 100.
    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces) # build with numpy.ndarray/torch.Tensor
    # torch.save(BVH, cubvh_path)
else:
    ### load BVH from file
    BVH = torch.load(cubvh_path)

### query ray-mesh intersection
# [N, 3], [N, 3], query with torch.Tensor (cuda)
rays_o, rays_d = torch.rand(1000, 3, device='cuda'), torch.rand(1000, 3, device='cuda')
rays_d = torch.nn.functional.normalize(rays_d, dim=-1) # normalize the ray directions
intersections, face_id, depth = BVH.ray_trace(rays_o, rays_d) # [N, 3], [N,], [N,]

### query unsigned distance
points = torch.rand(1000, 3, device='cuda') # [N, 3], query with torch.Tensor (cuda)
# uvw is the barycentric corrdinates of the closest point on the closest face (None if `return_uvw` is False).
distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True) # [N], [N], [N, 3]

### query signed distance (INNER is NEGATIVE!)
# for watertight meshes (default)
distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='watertight') # [N], [N], [N, 3]
# for non-watertight meshes:
distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='raystab') # [N], [N], [N, 3]

print(intersections.min(), intersections.max())
