from muv_convert.Module.step_loader import StepLoader

def demo():
    # step_file_path = "/Users/chli/chLi/Dataset/ABC/00000050_80d90bfdd2e74e709956122a_step_000.step"
    step_file_path = "/Users/chli/Downloads/FeiShu/JCD/111/11.step"

    step_loader = StepLoader()
    shape_data_list = step_loader.loadStepFile(step_file_path)

    if shape_data_list is None:
        print('loadStepFile failed!')
        return False

    step_loader.renderCADDataList(shape_data_list)

    for shape_data in shape_data_list:
        step_loader.renderCADData(shape_data)
    return True
