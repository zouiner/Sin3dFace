import numpy as np
import re
from pathlib import Path
from typing import Dict
from collections import defaultdict
from psbody.mesh import Mesh

import sys
sys.path.append('/users/ps1510/scratch/Programs/now_evaluation')

import scan2mesh_computations as s2m_opt
import scan2mesh_computations_metrical as s2m_opt_metrical

def load_pp(pp_path: str) -> np.ndarray:
    lmks = np.zeros((7, 3), dtype=np.float32)
    with open(pp_path, 'r') as f:
        lines = f.readlines()
        for j in range(8, 15):
            x = y = z = 0.0
            for val in lines[j].split(' '):
                if val.startswith("x="): x = float(val.split('"')[1])
                elif val.startswith("y="): y = float(val.split('"')[1])
                elif val.startswith("z="): z = float(val.split('"')[1])
            lmks[j - 8] = [x, y, z]
    return lmks

def load_txt(txt_path: str) -> np.ndarray:
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return np.array([list(map(float, line.split())) for line in lines], dtype=np.float32)

def compute_single_error(predicted_mesh_path: str,
                         predicted_landmark_path: str,
                         gt_mesh_path: str,
                         gt_lmk_path: str,
                         metrical_eval: bool = False,
                         unit: str = 'm') -> Dict[str, float]:
    pred_mesh = Mesh(filename=predicted_mesh_path)

    if predicted_landmark_path.endswith('.npy'):
        pred_lmks = np.load(predicted_landmark_path)
    else:
        pred_lmks = load_txt(predicted_landmark_path)

    gt_mesh = Mesh(filename=gt_mesh_path)
    gt_lmks = load_pp(gt_lmk_path)

    if metrical_eval:
        distances = s2m_opt_metrical.compute_errors(
            gt_mesh.v, gt_mesh.f, gt_lmks, pred_mesh.v, pred_mesh.f, pred_lmks,
            predicted_mesh_unit=unit, check_rigid_alignment=False)
    else:
        distances = s2m_opt.compute_errors(
            gt_mesh.v, gt_mesh.f, gt_lmks, pred_mesh.v, pred_mesh.f, pred_lmks,
            check_rigid_alignment=False)

    errors = np.hstack(distances)
    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors))
    }




# Example usage

now_index_path = '/users/ps1510/scratch/Dataset/Now_benchmark/imagepathsvalidation.txt'

# Build a lookup dictionary: {img_name (stem): [(subject, category, full_filename), ...]}
img_index = defaultdict(list)

with open(now_index_path, 'r') as f:
    for line in f:
        line = line.strip()
        path = Path(line)
        img_name = path.stem  # e.g. IMG_0053
        subject = path.parts[0]
        category = path.parts[1]
        img_index[img_name].append((subject, category, path.name))


path = '/users/ps1510/scratch/Programs/Sin3dFace/results/NoW_64_256_model1_s1_unkown_img/50000/3d_obj/01_IMG_0121'

name = Path(path).name
match = re.search(r'(IMG_\d+)', name)
img_id = []
if match:
    img_id.append(match.group(1))

for img in img_id:
    matches = img_index.get(img)

    for subject, category, fname in matches:

        result = compute_single_error(
            predicted_mesh_path= path + '/mesh.obj',
            predicted_landmark_path= path +'/kpt7.npy',
            gt_mesh_path='/users/ps1510/scratch/Dataset/Now_benchmark/scans/' + subject + '/natural_head_rotation.000001.obj',
            gt_lmk_path='/users/ps1510/scratch/Dataset/Now_benchmark/scans_lmks_onlypp/' + subject + '/natural_head_rotation.000001_picked_points.pp',
            metrical_eval=False,
            unit='m'
        )
        print(result)
