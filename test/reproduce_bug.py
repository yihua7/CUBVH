import os
import torch
import numpy as np
from cubvh.api import cuBVH


if __name__ == '__main__':
    root = '/mnt/pfs/data/huangyihua/AniGen_test/'
    instance = 'e8556485-7c18-4866-b92b-9000bc9aa8c9.e8556485-7c18-4866-b92b-9000bc9aa8c9.fbx_pose_3'
    skeleton_path = os.path.join(root, 'skeleton', instance, 'skeleton_voxelized.npz')
    skl_data = np.load(skeleton_path, allow_pickle=True)
    verts, face = np.array(skl_data['vertices'], dtype=np.float32), skl_data['faces']
    mesh = {
        "vertices" : torch.from_numpy(verts),
        "faces" : torch.from_numpy(face),
    }
    cubvh_path = os.path.join(root, 'skeleton', instance, 'cubvh.pth')
    if os.path.exists(cubvh_path):
        cubvh = torch.load(cubvh_path, weights_only=False)
    else:
        cubvh = cuBVH(mesh["vertices"], mesh["faces"], device="cpu")
        torch.save(cubvh, cubvh_path)
    
    samples = torch.randn(200_000, 3).cuda() * 2
    cubvh = cubvh.to(samples.device)
    udf, face_id, uvw = cubvh.unsigned_distance(samples, return_uvw=True)
    print(udf.max().item())
