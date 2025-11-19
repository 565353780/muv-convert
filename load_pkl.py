import pickle
import numpy as np

pkl_file_path = "/Users/chli/chLi/Dataset/ABC/pkl/00000050_80d90bfdd2e74e709956122a_step_000.pkl"

with open(pkl_file_path, "rb") as f:
    shape_data_list = pickle.load(f)

for shape_data in shape_data_list:
    data_type = shape_data['type']
    data = shape_data['data']

    print('==== shape ====')
    print('type:', data_type)

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
