import numpy as np
from typing import Union
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.compound import Compound
from occwl.uvgrid import ugrid, uvgrid
from occwl.entity_mapper import EntityMapper


def get_bbox(point_cloud):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return min_point, max_point

def update_mapping(data_dict):
    """
    移除未使用的索引键并重新映射
    """
    dict_new = {}
    mapping = {}
    if len(data_dict) == 0:
        return dict_new, mapping

    max_idx = max(data_dict.keys())
    skipped_indices = np.array(sorted(list(set(np.arange(max_idx)) - set(data_dict.keys()))))
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new] = value
        mapping[idx] = idx_new
    return dict_new, mapping


def face_edge_adj(shape: Union[Shell, Solid, Compound]):
    """
    从给定的shape中提取面/边几何信息并创建面-边邻接图

    Args:
        shape: Shell, Solid, 或 Compound对象

    Returns:
        face_dict: 面字典，面ID作为键
        edge_dict: 边字典，边ID作为键
        edgeFace_IncM: 边-面关联矩阵，边ID作为键，相邻面ID作为值
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)

    ### 提取面 ###
    face_dict = {}
    for face in shape.faces():
        face_idx = mapper.face_index(face)
        face_dict[face_idx] = (face.surface_type(), face)

    ### 提取边和关联矩阵 ###
    edgeFace_IncM = {}
    edge_dict = {}
    for edge in shape.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(shape.faces_from_edge(edge))
        if len(connected_faces) == 2 and not edge.seam(connected_faces[0]) and not edge.seam(connected_faces[1]):
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_idx = mapper.edge_index(edge) 
            edge_dict[edge_idx] = edge 
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)

            if edge_idx in edgeFace_IncM:
                edgeFace_IncM[edge_idx] += [left_index, right_index]
            else:
                edgeFace_IncM[edge_idx] = [left_index, right_index]
        else:
            pass  # 忽略seam边

    return face_dict, edge_dict, edgeFace_IncM

def extract_geometry_data(shape: Union[Shell, Solid, Compound], split_closed: bool=True) -> dict:
    """
    从shape中提取所有几何数据

    Args:
        shape: Shell, Solid, 或 Compound对象
        split_closed: 是否分割闭合面和闭合边

    Returns:
        data: 包含所有导出数据的字典
    """
    assert isinstance(shape, (Shell, Solid, Compound))

    # 分割闭合曲面和闭合曲线
    if split_closed:
        if isinstance(shape, Solid):
            shape = shape.split_all_closed_faces(num_splits=0)
            shape = shape.split_all_closed_edges(num_splits=0)
        # Shell和Compound也支持类似操作
        elif hasattr(shape, 'split_all_closed_faces'):
            shape = shape.split_all_closed_faces(num_splits=0)
            shape = shape.split_all_closed_edges(num_splits=0)

    # 提取面、边几何和面-边邻接关系
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(shape)

    # 跳过未使用的索引键，并更新邻接关系
    face_dict, face_map = update_mapping(face_dict)
    edge_dict, edge_map = update_mapping(edge_dict)
    edgeFace_IncM_update = {}
    for key, value in edgeFace_IncM.items():
        new_face_indices = [face_map[x] for x in value]
        edgeFace_IncM_update[edge_map[key]] = new_face_indices
    edgeFace_IncM = edgeFace_IncM_update

    # 构建面-边邻接关系
    num_faces = len(face_dict)
    if len(edgeFace_IncM) > 0:
        edgeFace_IncM_array = np.stack([x for x in edgeFace_IncM.values()])
    else:
        edgeFace_IncM_array = np.array([]).reshape(0, 2)

    faceEdge_IncM = []
    for surf_idx in range(num_faces):
        if len(edgeFace_IncM_array) > 0:
            surf_edges, _ = np.where(edgeFace_IncM_array == surf_idx)
            faceEdge_IncM.append(surf_edges)
        else:
            faceEdge_IncM.append(np.array([]))

    # 从曲面采样uv网格 (32x32)
    graph_face_feat = {}
    for face_idx, face_feature in face_dict.items():
        _, face = face_feature
        try:
            points = uvgrid(face, method="point", num_u=32, num_v=32)
            visibility_status = uvgrid(face, method="visibility_status", num_u=32, num_v=32)
            mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
            # 沿通道方向拼接形成面特征张量
            face_feat = np.concatenate((points, mask), axis=-1)
            graph_face_feat[face_idx] = face_feat
        except Exception as e:
            print(f"Warning: Failed to sample face {face_idx}: {e}")
            # 使用零填充
            graph_face_feat[face_idx] = np.zeros((32, 32, 4))

    if len(graph_face_feat) > 0:
        face_pnts = np.stack([x for x in graph_face_feat.values()])[:, :, :, :3]
    else:
        face_pnts = np.array([]).reshape(0, 32, 32, 3)

    # 从曲线采样u网格 (1x32)
    graph_edge_feat = {}
    graph_corner_feat = {}
    for edge_idx, edge in edge_dict.items():
        try:
            points = ugrid(edge, method="point", num_u=32)
            graph_edge_feat[edge_idx] = points
            # 边的起始/终止顶点
            v_start = points[0]
            v_end = points[-1]
            graph_corner_feat[edge_idx] = (v_start, v_end)
        except Exception as e:
            print(f"Warning: Failed to sample edge {edge_idx}: {e}")
            # 使用零填充
            graph_edge_feat[edge_idx] = np.zeros((32, 3))
            graph_corner_feat[edge_idx] = (np.zeros(3), np.zeros(3))

    if len(graph_edge_feat) > 0:
        edge_pnts = np.stack([x for x in graph_edge_feat.values()])
        edge_corner_pnts = np.stack([x for x in graph_corner_feat.values()])
    else:
        edge_pnts = np.array([]).reshape(0, 32, 3)
        edge_corner_pnts = np.array([]).reshape(0, 2, 3)

    data = {
        'face_pnts': face_pnts,
        'edge_pnts': edge_pnts,
        'edge_corner_pnts': edge_corner_pnts,
        'edgeFace_IncM': edgeFace_IncM_array,
        'faceEdge_IncM': faceEdge_IncM,
        'num_faces': num_faces,
        'num_edges': len(edge_dict)
    }
    return data
