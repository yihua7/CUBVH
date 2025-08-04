import trimesh
import torch
import numpy as np
from sklearn.decomposition import PCA
from cubvh import cuBVH


def get_transform(data):
    x = data.reshape([-1, data.shape[-1]])
    m, n = x.shape
    if n < 3:
        # Use normalization function if PCA cannot be applied
        bmax, bmin = x.max(dim=0).values, x.min(dim=0).values
        pad = 3 - n
        trans_func = lambda a: torch.cat([((a - bmin) / (bmax - bmin)).clamp(0., 1.), torch.ones((*a.shape[:-1], pad), device=a.device)], dim=-1)
        return trans_func
    else:
        U, _, V = torch.pca_lowrank(x, q=3)
        bmax, bmin = U.max(dim=0).values, U.min(dim=0).values
        trans_func = lambda a: ((torch.matmul(a, V) - bmin) / (bmax - bmin)).clamp(0., 1.)
        return trans_func


if __name__ == "__main__":
    mesh_recon = trimesh.load('../data/test/recon_mesh.obj', preprocess=False)
    data = np.load('../data/test/skeleton_voxelized.npz')
    skin = data['skin']
    skin = torch.tensor(skin, dtype=torch.float32).cuda()
    mesh_gt = trimesh.Trimesh(vertices=data['vertices'], faces=data['faces'], process=False)

    func = get_transform(skin)

    cubvh = cuBVH(mesh_gt.vertices, mesh_gt.faces)
    _, face_idx, uvw = cubvh.unsigned_distance(torch.tensor(mesh_recon.vertices, device='cuda:0'), return_uvw=True)
    face_idx = torch.tensor(data['faces']).cuda()[face_idx]
    skin_recon = (skin[face_idx] * uvw[..., None]).sum(dim=1)

    color_gt = func(skin)
    color_recon = func(skin_recon)

    # Save colored meshes using color_gt, color_recon
    gt_vertex_colors = (color_gt * 255).long().cpu().numpy()
    recon_vertex_colors = (color_recon * 255).long().cpu().numpy()

    mesh_gt = trimesh.Trimesh(vertices=mesh_gt.vertices, faces=mesh_gt.faces, vertex_colors=gt_vertex_colors)

    mesh_recon = trimesh.Trimesh(vertices=mesh_recon.vertices, faces= mesh_recon.faces, vertex_colors=recon_vertex_colors)

    mesh_gt.export('../data/test/mesh_skl_colored.obj')
    mesh_recon.export('../data/test/recon_mesh_colored.obj')

    # # Save colored point clouds
    # gt_point_cloud = trimesh.PointCloud(vertices=mesh_gt.vertices, colors=gt_vertex_colors)
    # recon_point_cloud = trimesh.PointCloud(vertices=mesh_recon.vertices, colors=recon_vertex_colors)
    # gt_point_cloud.export('../data/test/mesh_skl_colored.ply')
    # recon_point_cloud.export('../data/test/recon_mesh_colored.ply')

