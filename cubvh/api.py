import numpy as np
import torch

# CUDA extension
import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

class cuBVH():
    def __init__(self, vertices, triangles, bvh_nodes=None, triangles_data=None, device=None):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]
        # bvh_nodes: torch.Tensor, [N, 8], optional pre-computed BVH nodes
        # triangles_data: torch.Tensor, [M, 10], optional pre-computed sorted triangles
        # device: torch.device or str, device to create the BVH on

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        # check inputs
        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # Handle device specification
        if device is None:
            # Default to current CUDA device if available, otherwise cuda:0
            if torch.cuda.is_available():
                self._device = torch.cuda.current_device()
            else:
                raise RuntimeError("CUDA is not available")
        else:
            if isinstance(device, str):
                self._device = torch.device(device)
            else:
                self._device = device
            
            # Ensure it's a CUDA device
            if self._device.type != 'cuda':
                raise ValueError("cuBVH only supports CUDA devices")
        
        # Set the current CUDA device for BVH creation
        with torch.cuda.device(self._device):
            # Store original data for pickling
            self._vertices = vertices.copy()
            self._triangles = triangles.copy()
            
            # implementation
            self.impl = _backend.create_cuBVH(vertices, triangles)
            
            # If BVH nodes and triangles data are provided, set them directly instead of rebuilding
            if bvh_nodes is not None and triangles_data is not None:
                # Ensure data is on the correct device
                if bvh_nodes.device != self._device:
                    bvh_nodes = bvh_nodes.to(self._device)
                if triangles_data.device != self._device:
                    triangles_data = triangles_data.to(self._device)
                
                self.impl.set_bvh_nodes(bvh_nodes)
                self.impl.set_triangles(triangles_data)
                self._bvh_nodes = bvh_nodes
                self._triangles_data = triangles_data
            elif bvh_nodes is not None:
                # Only BVH nodes provided
                if bvh_nodes.device != self._device:
                    bvh_nodes = bvh_nodes.to(self._device)
                self.impl.set_bvh_nodes(bvh_nodes)
                self._bvh_nodes = bvh_nodes
                self._triangles_data = self.impl.get_triangles()
            else:
                # Get the BVH nodes and triangles that were built during construction
                self._bvh_nodes = self.impl.get_bvh_nodes()
                self._triangles_data = self.impl.get_triangles()

    @property
    def device(self):
        """Get the device this BVH is on."""
        return self._device
    
    def to(self, device, *args, **kwargs):
        """Move the BVH to a different device."""
        if isinstance(device, str):
            device = torch.device(device)
        
        if device.type != 'cuda':
            raise ValueError("cuBVH only supports CUDA devices")
        
        if device == self._device:
            return self  # Already on the target device
        
        # Create a new BVH on the target device using the saved BVH nodes and triangles
        with torch.cuda.device(device):
            # Create new implementation on target device
            new_impl = _backend.create_cuBVH_no_build(self._vertices, self._triangles)
            
            # Move BVH nodes and triangles to target device and set them
            bvh_nodes_on_device = self._bvh_nodes.to(device)
            triangles_data_on_device = self._triangles_data.to(device)
            new_impl.set_bvh_nodes(bvh_nodes_on_device)
            new_impl.set_triangles(triangles_data_on_device)
            
            # Create new cuBVH instance
            new_bvh = cuBVH.__new__(cuBVH)
            new_bvh._vertices = self._vertices.copy()
            new_bvh._triangles = self._triangles.copy()
            new_bvh._device = device
            new_bvh.impl = new_impl
            new_bvh._bvh_nodes = bvh_nodes_on_device
            new_bvh._triangles_data = triangles_data_on_device
            
            return new_bvh

    def ray_trace(self, rays_o, rays_d):
        # rays_o: torch.Tensor, float, [N, 3]
        # rays_d: torch.Tensor, float, [N, 3]

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        # Ensure inputs are on the same device as the BVH
        if rays_o.device != self._device:
            rays_o = rays_o.to(self._device)
        if rays_d.device != self._device:
            rays_d = rays_d.to(self._device)

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        with torch.cuda.device(self._device):
            # init output buffers
            positions = torch.empty(N, 3, dtype=torch.float32, device=self._device)
            face_id = torch.empty(N, dtype=torch.int64, device=self._device)
            depth = torch.empty(N, dtype=torch.float32, device=self._device)
            
            self.impl.ray_trace(rays_o, rays_d, positions, face_id, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        # Ensure input is on the same device as the BVH
        if positions.device != self._device:
            positions = positions.to(self._device)

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        with torch.cuda.device(self._device):
            # init output buffers
            distances = torch.empty(N, dtype=torch.float32, device=self._device)
            face_id = torch.empty(N, dtype=torch.int64, device=self._device)

            if return_uvw:
                uvw = torch.empty(N, 3, dtype=torch.float32, device=self._device)
            else:
                uvw = None
            
            self.impl.unsigned_distance(positions, distances, face_id, uvw) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    
    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        # Ensure input is on the same device as the BVH
        if positions.device != self._device:
            positions = positions.to(self._device)

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        with torch.cuda.device(self._device):
            # init output buffers
            distances = torch.empty(N, dtype=torch.float32, device=self._device)
            face_id = torch.empty(N, dtype=torch.int64, device=self._device)

            if return_uvw:
                uvw = torch.empty(N, 3, dtype=torch.float32, device=self._device)
            else:
                uvw = None
            
            self.impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode]) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    def get_bvh_nodes(self):
        """Get the BVH nodes as a tensor."""
        return self._bvh_nodes.clone()
    
    def get_triangles_data(self):
        """Get the sorted triangles data as a tensor."""
        return self._triangles_data.clone()
    
    def get_sorted_triangles(self):
        """Get the sorted triangles as vertices and triangle indices.
        
        Returns:
            vertices: torch.Tensor, [N, 3] - all vertex positions
            triangles: torch.Tensor, [M, 3] - triangle vertex indices  
            triangle_ids: torch.Tensor, [M] - original triangle IDs
        """
        triangles_data = self._triangles_data
        n_triangles = triangles_data.shape[0]
        
        # Extract vertices from triangles data (each triangle has 3 vertices)
        vertices = triangles_data[:, :9].view(n_triangles * 3, 3)  # [M*3, 3]
        triangle_ids = triangles_data[:, 9].long()  # [M]
        
        # Create triangle indices
        triangles = torch.arange(n_triangles * 3, device=self._device, dtype=torch.long).view(n_triangles, 3)
        
        return vertices, triangles, triangle_ids

    def free_memory(self):
        """Explicitly free GPU memory used by this BVH."""
        if hasattr(self, 'impl') and self.impl is not None:
            self.impl.free_memory()
    
    def __del__(self):
        """Destructor to ensure GPU memory is freed."""
        try:
            self.free_memory()
        except:
            # Ignore errors during destruction (e.g., if CUDA context is already destroyed)
            pass

    def __getstate__(self):
        # Return the state for pickling (exclude the C++ impl object)
        return {
            'vertices': self._vertices,
            'triangles': self._triangles,
            'bvh_nodes': self._bvh_nodes.cpu(),  # Save BVH nodes on CPU for portability
            'triangles_data': self._triangles_data.cpu(),  # Save sorted triangles on CPU for portability
            'device': self._device
        }
    
    def __setstate__(self, state):
        # Restore the state from pickling (rebuild the C++ impl object)
        self._vertices = state['vertices']
        self._triangles = state['triangles']
        self._device = state.get('device', torch.device('cuda:0'))  # Default to cuda:0 for backward compatibility
        
        # Move BVH nodes and triangles to the target device
        bvh_nodes = state['bvh_nodes'].to(self._device)
        triangles_data = state.get('triangles_data')  # May not exist in old saves
        self._bvh_nodes = bvh_nodes
        
        with torch.cuda.device(self._device):
            # Create the impl without building BVH, then set the saved data directly
            self.impl = _backend.create_cuBVH_no_build(self._vertices, self._triangles)
            self.impl.set_bvh_nodes(bvh_nodes)
            
            if triangles_data is not None:
                triangles_data = triangles_data.to(self._device)
                self.impl.set_triangles(triangles_data)
                self._triangles_data = triangles_data
            else:
                # For backward compatibility, get triangles from the current state
                self._triangles_data = self.impl.get_triangles()

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

def sparse_marching_cubes(coords, corners, iso):
    # coords: torch.Tensor, int32, [N, 3]
    # corners: torch.Tensor, float32, [N, 8]
    # iso: float

    coords = coords.int().contiguous()
    corners = corners.float().contiguous()

    if not coords.is_cuda: coords = coords.cuda()
    if not corners.is_cuda: corners = corners.cuda()

    verts, tris = _backend.sparse_marching_cubes(coords, corners, iso)

    return verts, tris