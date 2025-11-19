import os
from typing import Union

from muv_convert.Method.io import load_step_file, extract_all_shapes, parse_shape
from muv_convert.Method.render import vis_faces_edges, vis_faces_edges_list


class StepLoader(object):
    def __init__(self) -> None:
        return

    def loadStepFile(self, step_file_path: str) -> Union[list, None]:
        if not os.path.exists(step_file_path):
            print('[ERROR][StepLoader::loadStepFile]')
            print('\t step file not exist!')
            print('\t step_file_path:', step_file_path)
            return None

        shape = load_step_file(step_file_path)

        shapes_list = extract_all_shapes(shape)

        shape_data_list = []
        for shape_type, shape_obj in shapes_list:
            data = parse_shape(shape_obj, split_closed=True)
            shape_data_list.append({
                'type': shape_type,
                'data': data
            })

        return shape_data_list

    def renderCADData(self, shape_data: dict) -> bool:
        face_pts = shape_data['data']['face_pnts']
        edge_pts = shape_data['data']['edge_pnts']
        edge_corner_pts = shape_data['data']['edge_corner_pnts']

        vis_faces_edges(
            face_pts,
            edge_pts,
            edge_corner_pts,
        )
        return True

    def renderCADDataList(self, shape_data_list: list) -> bool:
        """
        可视化多个形状数据，所有形状在同一个窗口中显示

        Args:
            shape_data_list: 形状数据列表，每个元素是包含 'type' 和 'data' 的字典

        Returns:
            bool: 是否成功可视化
        """
        if len(shape_data_list) == 0:
            print('[ERROR][StepLoader::renderCADDataList]')
            print('\t shape_data_list is empty!')
            return False

        return vis_faces_edges_list(shape_data_list)
