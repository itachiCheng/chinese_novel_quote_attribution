import torch
class Configuration:
    rule_list = ["你", "我", "他", "你们", "我们", "他们", "您", "您们", "它", "它们", "她", "她们"]
    label_list = [0, 0, 0, 0, 0, 1, 0, 0, 1]
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
