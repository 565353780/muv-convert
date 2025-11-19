from muv_convert.Method.convert import igs_to_step

def demo():
    igs_file_path = "/Users/chli/Downloads/FeiShu/JCD/111/11.igs"
    step_file_path = "/Users/chli/Downloads/FeiShu/JCD/111/11.step"
    overwrite = True

    igs_to_step(igs_file_path, step_file_path, overwrite)
    return True
