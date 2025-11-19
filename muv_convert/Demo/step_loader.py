from muv_convert.Module.step_loader import StepLoader

def demo():
    step_file_path = "/Users/chli/chLi/Dataset/ABC/00000050_80d90bfdd2e74e709956122a_step_000.step"
    step_file_path = "/Users/chli/Downloads/FeiShu/JCD/111/11.step"

    step_loader = StepLoader()
    cad_data_list = step_loader.loadStepFile(step_file_path)

    if cad_data_list is None:
        print('loadStepFile failed!')
        return False

    for cad_data in cad_data_list:
        step_loader.renderCADData(cad_data)
    return True
