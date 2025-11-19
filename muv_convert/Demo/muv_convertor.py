from muv_convert.Module.muv_convertor import MUVConvertor

def demo():
    step_file_path = "/Users/chli/chLi/Dataset/ABC/00000050_80d90bfdd2e74e709956122a_step_000.step"
    save_pkl_file_path = "/Users/chli/chLi/Dataset/ABC/pkl/00000050_80d90bfdd2e74e709956122a_step_000.pkl"
    overwrite = True

    muv_convertor = MUVConvertor()
    muv_convertor.convertStepFile(
        step_file_path,
        save_pkl_file_path,
        overwrite
    )
    return True
