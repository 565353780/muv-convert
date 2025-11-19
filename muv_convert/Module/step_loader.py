import os
from typing import Union
from occwl.io import load_step

from muv_convert.Method.io import parse_solid
from muv_convert.Method.render import vis_faces_edges


class StepLoader(object):
    def __init__(self) -> None:
        return

    def loadStepFile(self, step_file_path: str) -> Union[list, None]:
        if not os.path.exists(step_file_path):
            print('[ERROR][StepLoader::loadStepFile]')
            print('\t step file not exist!')
            print('\t step_file_path:', step_file_path)
            return None

        cad_solid_list = load_step(step_file_path)

        if len(cad_solid_list) == 0:
            print('[WARN][StepLoader::loadStepFile]')
            print('\t cad solid not found!')
            print('\t step_file_path:', step_file_path)
            return None

        cad_data_list = []

        for cad_solid in cad_solid_list:
            cad_data = parse_solid(cad_solid)
            cad_data_list.append(cad_data)

        return cad_data_list

    def renderCADData(self, cad_data: dict) -> bool:
        vis_faces_edges(
            cad_data['surf_wcs'],
            cad_data['edge_wcs'],
            cad_data['corner_wcs'],
        )
        return True
