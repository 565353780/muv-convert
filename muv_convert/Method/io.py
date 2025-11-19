import numpy as np
from typing import Union
from occwl.solid import Solid

from muv_convert.Config.constant import MAX_FACE
from muv_convert.Method.convert_utils import (
    extract_primitive,
    get_bbox,
)
from muv_convert.Method.transform import normalize


def parse_solid(solid: Solid) -> Union[dict, None]:
    """
    Parse the surface, curve, face, edge, vertex in a CAD solid.

    Args:
    - solid (occwl.solid): A single brep solid in occwl data format.

    Returns:
    - data: A dictionary containing all parsed data
    """
    assert isinstance(solid, Solid)

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    if MAX_FACE > 0:
        if len(list(solid.faces())) > MAX_FACE:
            return None

    # Extract all B-rep primitives and their adjacency information
    face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM = extract_primitive(solid)

    # Normalize the CAD model
    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs = normalize(face_pnts, edge_pnts, edge_corner_pnts)

    # Remove duplicate and merge corners
    corner_wcs = np.round(corner_wcs,4)
    corner_unique = []
    for corner_pnt in corner_wcs.reshape(-1,3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1,3)
        else:
            # Check if it exist or not
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if exists:
                continue
            else:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1,3)], 0)
    corner_unique = np.asarray(corner_unique)

    # Edge-corner adjacency
    edgeCorner_IncM = []
    for edge_corner in corner_wcs:
        start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
        end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
        edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM)

    # Surface global bbox
    surf_bboxes = []
    for pnts in surfs_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1,3))
        surf_bboxes.append( np.concatenate([min_point, max_point]))
    surf_bboxes = np.vstack(surf_bboxes)

    # Edge global bbox
    edge_bboxes = []
    for pnts in edges_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1,3))
        edge_bboxes.append(np.concatenate([min_point, max_point]))
    edge_bboxes = np.vstack(edge_bboxes)

    # Convert to float32 to save space
    data = {
        'surf_wcs':surfs_wcs.astype(np.float32),
        'edge_wcs':edges_wcs.astype(np.float32),
        'surf_ncs':surfs_ncs.astype(np.float32),
        'edge_ncs':edges_ncs.astype(np.float32),
        'corner_wcs':corner_wcs.astype(np.float32),
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj':edgeCorner_IncM,
        'faceEdge_adj':faceEdge_IncM,
        'surf_bbox_wcs':surf_bboxes.astype(np.float32),
        'edge_bbox_wcs':edge_bboxes.astype(np.float32),
        'corner_unique':corner_unique.astype(np.float32),
    }

    return data
