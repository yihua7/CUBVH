from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image

        return pack
    

class AniGenSparseFeat2Skeleton(StandardDatasetBase):
    """
    SparseFeat2Render dataset.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        load_cubvh: bool = False,
        skl_dilation_iter: int = 0,
        filter_bad_skin: bool = False,

        test_mode: bool = True,  # Test the model performance
        is_test: bool = False,  # Train or validation
        skin_accum_as_flow: bool = False,  # Accumulate skin weights from bottom to top as flow-by probability

        local_rank: int = 0
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.load_cubvh = load_cubvh
        self.skl_dilation_iter = skl_dilation_iter
        self.filter_bad_skin = filter_bad_skin

        self.test_mode = test_mode
        self.is_test = is_test
        self.skin_accum_as_flow = skin_accum_as_flow
        self.local_rank = local_rank

        super().__init__(roots)
        self.is_bad_skin_list = self.metadata['is_bad_skin'].values
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'feature_{self.model}']]
        stats['With features'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)

        if 'is_bad_skeleton' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skeleton']]
        if self.filter_bad_skin and 'is_bad_skin' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skin']]

        if self.test_mode:
            metadata = metadata[metadata['is_test']] if self.is_test else metadata[~metadata['is_test']]

        return metadata, stats
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        # collate other data
        keys = [k for k in batch[0].keys() if k not in ['coords', 'feats', 'coords_skl', 'feats_skl', 'image', 'alpha', 'extrinsics', 'intrinsics', 'joints', 'parents', 'skin']]
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                pack[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], list):
                pack[k] = sum([b[k] for b in batch], [])
            else:
                pack[k] = [b[k] for b in batch]

        return pack
        
    def _get_geo(self, root, instance):
        from cubvh import cuBVH
        torch.cuda.set_device(self.local_rank)
        skeleton_path = os.path.join(root, 'skeleton', instance, 'skeleton_voxelized.npz')
        skl_data = np.load(skeleton_path, allow_pickle=True)
        verts, face = np.array(skl_data['vertices'], dtype=np.float32), skl_data['faces']
        mesh = {
            "vertices" : torch.from_numpy(verts),
            "faces" : torch.from_numpy(face),
        }
        geo = {"mesh": mesh}
        if self.load_cubvh:
            if cuBVH is None:
                raise RuntimeError("cuBVH is required when load_cubvh is True.")
            cubvh_path = os.path.join(root, 'skeleton', instance, 'cubvh.pth')
            # Debug!!!
            if os.path.exists(cubvh_path) and False:  # Debug!!!
                cubvh = torch.load(cubvh_path, weights_only=False)
                if isinstance(cubvh, cuBVH):
                    cubvh = cubvh.to('cpu')
            else:
                device = torch.device(f"cuda:{self.local_rank}")
                cubvh = cuBVH(mesh["vertices"], mesh["faces"], device=device)
                cubvh = cubvh.to('cpu')
                # torch.save(cubvh, cubvh_path)  # Debug!!!
            geo["cubvh"] = cubvh
        return  geo

    def get_instance(self, root, instance):
        geo = self._get_geo(root, instance)
        
        return {
            **geo,
            'instance': instance,
        }
