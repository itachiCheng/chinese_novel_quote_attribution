from tageditor import DiaTextProcess

if __name__ == "__main__":
    model = "./model/"
    path = "./txt_data/"
    output_path = "./output_data/"
    tagpreprocess = DiaTextProcess(path, model, output_path)
    tagpreprocess.generate_tag()
