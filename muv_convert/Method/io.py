import numpy as np
from typing import Union
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.compound import Compound
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL
from OCC.Core.TopoDS import topods_Solid, topods_Shell

from muv_convert.Method.convert_utils import (
    get_bbox,
    extract_geometry_data,
)


def load_step_file(step_file_path: str):
    """
    使用pythonocc-core读取STEP文件，返回所有形状

    Args:
        step_file_path: STEP文件路径

    Returns:
        shapes: 所有形状的列表
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)

    if status != IFSelect_RetDone:
        print('[ERROR][io::loadStepFile]')
        print(f"\t Error reading STEP file: {step_file_path}")
        return None

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    return shape

def extract_all_shapes(shape):
    """
    从顶层shape中提取所有可处理的形状（Solid, Shell, Compound）

    Args:
        shape: OCC TopoDS_Shape对象

    Returns:
        shapes_list: 包含所有可处理形状的列表
    """
    shapes_list = []

    # 尝试提取Solid
    exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp_solid.More():
        solid = topods_Solid(exp_solid.Current())
        try:
            occwl_solid = Solid(solid)
            shapes_list.append(('Solid', occwl_solid))
        except Exception as e:
            print(f"Warning: Failed to convert Solid: {e}")
        exp_solid.Next()

    # 如果没有Solid，尝试提取Shell
    if len(shapes_list) == 0:
        exp_shell = TopExp_Explorer(shape, TopAbs_SHELL)
        while exp_shell.More():
            shell = topods_Shell(exp_shell.Current())
            try:
                occwl_shell = Shell(shell)
                shapes_list.append(('Shell', occwl_shell))
            except Exception as e:
                print(f"Warning: Failed to convert Shell: {e}")
            exp_shell.Next()

    # 如果还是没有，尝试将整个shape作为Compound处理
    if len(shapes_list) == 0:
        try:
            occwl_compound = Compound(shape)
            shapes_list.append(('Compound', occwl_compound))
        except Exception as e:
            print(f"Warning: Failed to convert Compound: {e}")

    return shapes_list


def parse_shape(shape_obj: Union[Shell, Solid, Compound], split_closed: bool = True) -> dict:
    """
    从shape中提取原始几何数据，不进行归一化处理

    Args:
        shape_obj: Shell, Solid, 或 Compound对象
        split_closed: 是否分割闭合面和闭合边

    Returns:
        data: A dictionary containing all parsed data
            - face_pnts: (N, 32, 32, 4) 面采样点和mask，前3维是xyz坐标，第4维是mask
            - edge_pnts: (M, 32, 3) 边采样点
            - edge_corner_pnts: (M, 2, 3) 边的起始和终止顶点
            - edgeFace_IncM: (M, 2) 边-面邻接矩阵
            - faceEdge_IncM: list of arrays，面-边邻接关系
            - corner_unique: 去重后的顶点
            - edgeCorner_IncM: 边-顶点邻接关系
    """

    data = extract_geometry_data(shape_obj, split_closed)

    face_pnts = data['face_pnts']  # (N, 32, 32, 4) - 包含xyz和mask
    edge_pnts = data['edge_pnts']  # (M, 32, 3)
    edge_corner_pnts = data['edge_corner_pnts']  # (M, 2, 3)
    edgeFace_IncM = data['edgeFace_IncM']
    faceEdge_IncM = data['faceEdge_IncM']

    # Remove duplicate and merge corners
    corner_pnts = np.round(edge_corner_pnts, 4)
    corner_unique = []
    for corner_pnt in corner_pnts.reshape(-1, 3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1, 3)
        else:
            # Check if it exists or not
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if not exists:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1, 3)], 0)
    corner_unique = np.asarray(corner_unique) if len(corner_unique) > 0 else np.array([]).reshape(0, 3)

    # Edge-corner adjacency
    edgeCorner_IncM = []
    if corner_unique.size > 0 and corner_pnts.size > 0:
        for edge_corner in corner_pnts:
            start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
            end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
            edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM) if len(edgeCorner_IncM) > 0 else np.array([]).reshape(0, 2)

    # Convert to float32 to save space
    result_data = {
        # 原始几何数据（不归一化）
        'face_pnts': face_pnts.astype(np.float32),  # (N, 32, 32, 4) 包含xyz和mask
        'edge_pnts': edge_pnts.astype(np.float32),  # (M, 32, 3)
        'edge_corner_pnts': edge_corner_pnts.astype(np.float32),  # (M, 2, 3)

        # 邻接关系
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj': edgeCorner_IncM,
        'faceEdge_adj': faceEdge_IncM,

        # 顶点
        'corner_unique': corner_unique.astype(np.float32),
    }

    return result_data
