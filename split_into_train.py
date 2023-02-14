from tageditor import DiaTextProcess

if __name__ == "__main__":
    modify_data = "./modify_data/"
    output_train = "./modify_output_data/"
    DiaTextProcess.generate_train_sample(modify_data, output_train)
