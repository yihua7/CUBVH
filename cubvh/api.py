import contextlib
import numpy as np
import torch

# CUDA extension
import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

class cuBVH:
    def __init__(self, vertices=None, triangles=None, *, state=None, device=None):
        if state is None and (vertices is None or triangles is None):
            raise ValueError("cuBVH requires either vertices/triangles or a serialized state.")

        self._impl = None
        self._impl_device = None
        self._state = None
        self.device = torch.device('cpu')

        target_device = self._parse_device(device)

        if state is not None:
            self._load_state(state)
        else:
            self._build_from_mesh(vertices, triangles, target_device)

        try:
            self.to(target_device)
        except RuntimeError:
            if target_device.type != 'cpu':
                self.to(torch.device('cpu'))
            else:
                raise

    @staticmethod
    def _parse_device(device):
        if device is None:
            return torch.device('cpu')
        return torch.device(device)

    def _build_from_mesh(self, vertices, triangles, build_device):
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles):
            triangles = triangles.detach().cpu().numpy()

        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."

        if build_device.type != 'cuda':
            tri_pos, tri_ids, node_mins, node_maxs, node_children = _backend.build_cuBVH_state(vertices, triangles)
            self._state = {
                "triangles": tri_pos.detach().clone().cpu().contiguous(),
                "triangle_ids": tri_ids.detach().clone().cpu().contiguous(),
                "node_mins": node_mins.detach().clone().cpu().contiguous(),
                "node_maxs": node_maxs.detach().clone().cpu().contiguous(),
                "node_children": node_children.detach().clone().cpu().contiguous(),
            }
            self._impl = None
            self._impl_device = None
            self.device = torch.device('cpu')
            return

        with self._cuda_device_guard(build_device):
            self._impl = _backend.create_cuBVH(vertices, triangles)
            self._impl_device = build_device
            self.device = build_device
            self._state = self._pull_state_from_impl()

    def _pull_state_from_impl(self):
        with self._cuda_device_guard(self._impl_device):
            tri_pos, tri_ids, node_mins, node_maxs, node_children = self._impl.export_state()
        return {
            "triangles": tri_pos.detach().clone().cpu().contiguous(),
            "triangle_ids": tri_ids.detach().clone().cpu().contiguous(),
            "node_mins": node_mins.detach().clone().cpu().contiguous(),
            "node_maxs": node_maxs.detach().clone().cpu().contiguous(),
            "node_children": node_children.detach().clone().cpu().contiguous(),
        }

    def _load_state(self, state):
        required = {"triangles", "triangle_ids", "node_mins", "node_maxs", "node_children"}
        missing = required.difference(state.keys())
        if missing:
            raise ValueError(f"Serialized state is missing keys: {sorted(missing)}")

        self._state = {
            "triangles": torch.as_tensor(state["triangles"], dtype=torch.float32, device='cpu').contiguous().clone(),
            "triangle_ids": torch.as_tensor(state["triangle_ids"], dtype=torch.long, device='cpu').contiguous().clone(),
            "node_mins": torch.as_tensor(state["node_mins"], dtype=torch.float32, device='cpu').contiguous().clone(),
            "node_maxs": torch.as_tensor(state["node_maxs"], dtype=torch.float32, device='cpu').contiguous().clone(),
            "node_children": torch.as_tensor(state["node_children"], dtype=torch.int32, device='cpu').contiguous().clone(),
        }

        self._impl = None
        self._impl_device = None
        self.device = torch.device('cpu')

    def _release_impl(self):
        if self._impl is None:
            return

        if self._impl_device is not None and self._impl_device.type == 'cuda':
            with self._cuda_device_guard(self._impl_device):
                torch.cuda.synchronize()

        self._impl = None
        self._impl_device = None

    def _instantiate_impl(self, cuda_device):
        triangles = self._state["triangles"].contiguous()
        triangle_ids = self._state["triangle_ids"].contiguous()
        node_mins = self._state["node_mins"].contiguous()
        node_maxs = self._state["node_maxs"].contiguous()
        node_children = self._state["node_children"].contiguous()

        with self._cuda_device_guard(cuda_device):
            self._impl = _backend.create_cuBVH_from_state(
                triangles,
                triangle_ids,
                node_mins,
                node_maxs,
                node_children,
            )

        self._impl_device = cuda_device
        self.device = cuda_device

    def to(self, device):
        device = self._parse_device(device)

        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for cuBVH.to().")
            index = device.index if device.index is not None else torch.cuda.current_device()
            cuda_device = torch.device('cuda', index)
            if self._impl is not None and self._impl_device == cuda_device:
                self.device = cuda_device
                return self
            self._release_impl()
            self._instantiate_impl(cuda_device)
        elif device.type == 'cpu':
            self._release_impl()
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"Unsupported device for cuBVH: {device}")

        return self

    def _require_impl(self):
        if self._impl is None:
            raise RuntimeError("cuBVH is on CPU; call cuBVH.to('cuda') before querying.")
        return self._impl

    def ray_trace(self, rays_o, rays_d):
        impl = self._require_impl()
        if self.device.type != 'cuda':
            raise RuntimeError("cuBVH must be on a CUDA device to trace rays.")

        target_device = self.device
        rays_o = rays_o.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()
        rays_d = rays_d.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        n_rays = rays_o.shape[0]

        with self._cuda_device_guard(target_device):
            positions = torch.empty(n_rays, 3, dtype=torch.float32, device=target_device)
            face_id = torch.empty(n_rays, dtype=torch.int64, device=target_device)
            depth = torch.empty(n_rays, dtype=torch.float32, device=target_device)

            impl.ray_trace(rays_o, rays_d, positions, face_id, depth)

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        impl = self._require_impl()
        if self.device.type != 'cuda':
            raise RuntimeError("cuBVH must be on a CUDA device to query distances.")

        target_device = self.device
        positions = positions.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        n_points = positions.shape[0]

        with self._cuda_device_guard(target_device):
            distances = torch.empty(n_points, dtype=torch.float32, device=target_device)
            face_id = torch.empty(n_points, dtype=torch.int64, device=target_device)

            if return_uvw:
                uvw = torch.empty(n_points, 3, dtype=torch.float32, device=target_device)
            else:
                uvw = None

            impl.unsigned_distance(positions, distances, face_id, uvw)

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        impl = self._require_impl()
        if self.device.type != 'cuda':
            raise RuntimeError("cuBVH must be on a CUDA device to query distances.")

        target_device = self.device
        positions = positions.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        n_points = positions.shape[0]

        with self._cuda_device_guard(target_device):
            distances = torch.empty(n_points, dtype=torch.float32, device=target_device)
            face_id = torch.empty(n_points, dtype=torch.int64, device=target_device)

            if return_uvw:
                uvw = torch.empty(n_points, 3, dtype=torch.float32, device=target_device)
            else:
                uvw = None

            impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode])

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    def export_state(self):
        return {key: value.clone() for key, value in self._state.items()}

    @property
    def triangles_cpu(self):
        return self._state["triangles"].detach().cpu().clone().numpy()

    @property
    def bvh_nodes_cpu(self):
        return {
            "mins": self._state["node_mins"].detach().cpu().clone().numpy(),
            "maxs": self._state["node_maxs"].detach().cpu().clone().numpy(),
            "children": self._state["node_children"].detach().cpu().clone().numpy(),
        }

    def __getstate__(self):
        return {
            "state": self.export_state(),
            "device": str(self.device),
        }

    def __setstate__(self, state):
        self.__init__(state=state["state"], device=state.get("device", "cpu"))

    @contextlib.contextmanager
    def _cuda_device_guard(self, device):
        if device is None or device.type != 'cuda':
            yield
        else:
            with torch.cuda.device(device):
                yield

def floodfill(grid):
    # grid: torch.Tensor, uint8, [B, H, W, D] or [H, W, D]
    # return: torch.Tensor, int32, [B, H, W, D] or [H, W, D], label of the connected component (value can be 0 to H*W*D-1, not remapped!)

    grid = grid.contiguous()
    if not grid.is_cuda: grid = grid.cuda()

    if grid.dim() == 3:
        mask = _backend.floodfill(grid.unsqueeze(0)).squeeze(0)
    else:
        mask = _backend.floodfill(grid)

    return mask


class cuHashTable:
    """
    Python wrapper around the CUDA ND integer hash table.

    - Default dimensionality is 3; can be changed via num_dims argument or set_num_dims.
    - Static table: prefer a single build() call; repeated insert() calls overwrite indices.
    """

    def __init__(self, num_dims: int = 3):
        # create implementation via factory (mirrors cuBVH style)
        self.impl = _backend.create_cuHashTable()
        self.impl.set_num_dims(int(num_dims))

    @property
    def num_dims(self) -> int:
        return int(self.impl.get_num_dims())

    def build(self, coords):
        """Build table from coordinates: coords [N,D] int32/cuda.
        Auto-sets capacity to max(16, 2*N)."""
        if coords.shape[1] != self.num_dims:
            self.impl.set_num_dims(int(coords.size(1)))
        self.impl.build(coords)

    def search(self, queries) -> torch.Tensor:
        """Search queries [M,D] -> indices [M] int32 on CUDA; -1 if not found."""
        assert queries.shape[1] == self.num_dims, f"queries must be {self.num_dims}D"
        return self.impl.search(queries)

    
def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):
    # coords: torch.Tensor, int32, [N, 3]
    # corners: torch.Tensor, float32, [N, 8]
    # iso: float
    # ensure_consistency: bool, whether to ensure shared corner values are consistent

    coords = coords.int().contiguous()
    corners = corners.float().contiguous()

    if not coords.is_cuda: coords = coords.cuda()
    if not corners.is_cuda: corners = corners.cuda()

    verts, tris = _backend.sparse_marching_cubes(coords, corners, iso, ensure_consistency)

    return verts, tris

# CPU hole filling numpy API
def fill_holes(vertices: np.ndarray, faces: np.ndarray, return_added: bool = False, check_containment: bool = True, eps: float = 1e-7, verbose: bool = False) -> np.ndarray:
    """
    Fill small holes in a triangular mesh using a CPU ear-clipping strategy.

    Args:
        vertices (np.ndarray float32 [N,3])
        faces (np.ndarray int32 [M,3])
        return_added: if True, return only newly added triangles; else full face list
        check_containment: avoid creating triangles containing other boundary verts
        eps: numeric epsilon
        verbose: print detailed logs from C++
    Returns:
        np.ndarray int32 [...,3]
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    faces = _backend.fill_holes(vertices, faces, return_added, check_containment, float(eps), bool(verbose))
    return np.asarray(faces, dtype=np.int32)

def merge_vertices(vertices: np.ndarray, faces: np.ndarray, threshold: float = 1e-3):
    """Merge vertices closer than threshold.
    Args:
        vertices (np.ndarray float32 [N,3])
        faces (np.ndarray int32 [M,3])
        threshold (float): distance threshold
    Returns:
        (vertices, faces) after merging
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    assert vertices.ndim==2 and vertices.shape[1]==3
    assert faces.ndim==2 and faces.shape[1]==3
    v_new, f_new = _backend.merge_vertices(vertices, faces, float(threshold))
    return np.asarray(v_new, dtype=np.float32), np.asarray(f_new, dtype=np.int32)


class HashTable:
    """
    CPU ND integer hash table (static, open-addressed). Mirrors cuHashTable but on host.
    """
    def __init__(self, num_dims: int = 3):
        # constructed directly from backend class
        self.impl = _backend.HashTable()
        self.impl.set_num_dims(int(num_dims))

    @property
    def num_dims(self) -> int:
        return int(self.impl.get_num_dims())

    def build(self, coords):
        """Build table from coordinates: coords [N,D] int32/CPU.
        Auto-sets capacity to max(16, 2*N)."""
        if coords.shape[1] != self.num_dims:
            self.impl.set_num_dims(int(coords.shape[1]))
        coords = coords.int().contiguous().cpu()
        self.impl.build(coords)

    def search(self, queries):
        """Search queries [M,D] -> indices [M] int32/CPU; -1 if not found."""
        assert queries.shape[1] == self.num_dims, f"queries must be {self.num_dims}D"
        queries = queries.int().contiguous().cpu()
        return self.impl.search(queries)

def sparse_marching_cubes_cpu(coords, corners, iso: float, ensure_consistency: bool = False):
    """CPU sparse marching cubes wrapper.
    Args:
        coords: (N,3) int32 voxel coordinates (torch.Tensor or np.ndarray)
        corners: (N,8) float32 corner SDF values (torch.Tensor or np.ndarray)
        iso: isovalue
        ensure_consistency: average shared corners across voxels before extraction
    Returns:
        (vertices, faces): np.ndarray float32 [M,3], np.ndarray int32 [T,3]
    """
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    if torch.is_tensor(corners):
        corners = corners.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=np.int32)
    corners = np.asarray(corners, dtype=np.float32)
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be [N,3]"
    assert corners.ndim == 2 and corners.shape[1] == 8, "corners must be [N,8]"
    v, f = _backend.sparse_marching_cubes_cpu(coords, corners, float(iso), bool(ensure_consistency))
    return np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32)


def decimate(vertices: np.ndarray, faces: np.ndarray, target_vertices: int):
    """CPU quadric-error simplification to target number of vertices.
    Args:
        vertices: np.ndarray float32 or float64 [N,3]
        faces: np.ndarray int32 [M,3]
        target_vertices: desired vertex count after decimation
    Returns:
        (vertices, faces): simplified mesh
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    faces = faces.astype(np.int32)
    v, f = _backend.decimate(vertices, faces, int(target_vertices))
    return v, f


def parallel_decimate(vertices: np.ndarray, faces: np.ndarray, target_vertices: int):
    """CPU batch-parallel decimation to target number of vertices.
    Args:
        vertices: np.ndarray float32 or float64 [N,3]
        faces: np.ndarray int32 [M,3]
        target_vertices: desired vertex count after decimation
    Returns:
        (vertices, faces): simplified mesh
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    faces = faces.astype(np.int32)
    v, f = _backend.parallel_decimate(vertices, faces, int(target_vertices))
    return v, f