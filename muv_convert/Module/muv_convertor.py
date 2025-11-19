import os
import pickle 

from muv_convert.Method.path import createFileFolder, removeFile
from muv_convert.Module.step_loader import StepLoader

class MUVConvertor(StepLoader):
    def __init__(self) -> None:
        StepLoader.__init__(self)
        return

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

        with open(save_pkl_file_path, "wb") as tf:
            pickle.dump(cad_data_list, tf)
        return True
