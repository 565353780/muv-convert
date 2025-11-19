import os
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static

from muv_convert.Method.path import createFileFolder, removeFile


def igs_to_step(
    igs_file_path: str,
    save_step_file_path: str,
    overwrite: bool = False,
) -> bool:
    if not os.path.exists(igs_file_path):
        print('[ERROR][convert::igs_to_step]')
        print('\t igs file not exist!')
        print('\t igs_file_path:', igs_file_path)
        return False

    if os.path.exists(save_step_file_path):
        if not overwrite:
            return True

        removeFile(save_step_file_path)

    # Load IGES file
    iges_reader = IGESControl_Reader()
    status = iges_reader.ReadFile(igs_file_path)

    if status != 1:
        print('[ERROR][convert::igs_to_step]')
        print(f"\t IGES file read error: status {status}")
        return False

    # Transfer all roots (import entire model)
    iges_reader.TransferRoots()

    # Retrieve the resulting shape
    shape = iges_reader.OneShape()

    # STEP writer
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")  # or AP214

    step_writer.Transfer(shape, STEPControl_AsIs)

    createFileFolder(save_step_file_path)

    status = step_writer.Write(save_step_file_path)

    if status != 1:
        print('[ERROR][convert::igs_to_step]')
        print(f"\t Failed to write STEP: status {status}")
        return False

    return True
