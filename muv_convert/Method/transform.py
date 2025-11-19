import numpy as np


def normalize(surf_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization 
    """
    # Global normalization to -1~1
    total_points = np.array(surf_pnts).reshape(-1, 3)
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals)/2 
    global_scale = max(max_vals - min_vals)
    assert global_scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [],[],[],[]

    # Normalize corner
    corner_wcs = (corner_pnts - global_offset[np.newaxis,:]) / (global_scale * 0.5)

    # Normalize surface
    for surf_pnt in surf_pnts:
        # Normalize CAD to WCS
        surf_pnt_wcs = (surf_pnt - global_offset[np.newaxis,np.newaxis,:]) / (global_scale * 0.5)
        surfs_wcs.append(surf_pnt_wcs)
        # Normalize Surface to NCS
        min_vals = np.min(surf_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(surf_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis,np.newaxis,:]) / (local_scale * 0.5)
        surfs_ncs.append(pnt_ncs)

    # Normalize edge
    for edge_pnt in edge_pnts:
        # Normalize CAD to WCS
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis,:]) / (global_scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        # Normalize Edge to NCS
        min_vals = np.min(edge_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs
