import os
import pickle 
from typing import Union
from occwl.io import load_step

from muv_convert.Method.io import parse_solid
from muv_convert.Method.path import createFileFolder, removeFile

class MUVConvertor(object):
    def __init__(self) -> None:
        return

    def loadStepFile(self, step_file_path: str) -> Union[list, None]:
        if not os.path.exists(step_file_path):
            print('[ERROR][MUVConvertor::loadStepFile]')
            print('\t step file not exist!')
            print('\t step_file_path:', step_file_path)
            return None

        cad_solid_list = load_step(step_file_path)

        if len(cad_solid_list) == 0:
            print('[WARN][MUVConvertor::loadStepFile]')
            print('\t cad solid not found!')
            print('\t step_file_path:', step_file_path)
            return None

        cad_data_list = []

        for cad_solid in cad_solid_list:
            cad_data = parse_solid(cad_solid)
            cad_data_list.append(cad_data)

        return cad_data_list

    def convertStepFile(
        self,
        step_file_path: str,
        save_pkl_file_path: str,
        overwrite: bool = False,
    ) -> bool:
        if os.path.exists(save_pkl_file_path):
            if not overwrite:
                return True

            removeFile(save_pkl_file_path)

        cad_data_list = self.loadStepFile(step_file_path)

        if cad_data_list is None:
            print('[ERROR][MUVConvertor::convertStepFile]')
            print('\t loadStepFile failed!')
            return False

        createFileFolder(save_pkl_file_path)

        for i, cad_data in enumerate(cad_data_list):
            curr_save_pkl_file_path = save_pkl_file_path[:-4] + '_' + str(i) + '.pkl'
            with open(curr_save_pkl_file_path, "wb") as tf:
                pickle.dump(cad_data, tf)
        return True
