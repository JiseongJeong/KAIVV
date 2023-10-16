import os

def create_pathlist(data_folder):
    file_paths = []
    file_list = os.listdir(data_folder)
    for file_name in file_list :
        file_path = os.path.join(data_folder, file_name)
        file_paths.append(file_path)
    return file_paths
