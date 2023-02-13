#!/usr/bin/python3
# coding:utf-8

import os
import re
import json
from collections import defaultdict
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from configs import Configuration


class TextProcess(object):
    r"""General Chinese Text Process

        This is a module provided common methods in processing Chinese Text.

        Args:

            in_features: files_path
    """

    def __init__(self, files_path):
        self.files_list = []

        files_gener = os.walk(files_path)

        for root, dirs, files in files_gener:
            for file_name in files:
                if not file_name.endswith(".txt"):
                    continue
                txt_path = os.path.join(root, file_name)
                self.files_list.append(txt_path)

    @staticmethod
    def cut_sentence(txt_str):
        txt_str = txt_str.replace("\r", "").replace("\n", "")
        txt_str = re.sub("([；;。？！])([^”’])", r"\1\n\2", txt_str)
        txt_str = re.sub("(\…{2})([^”’])", r"\1\n\2", txt_str)
        txt_str = re.sub("(\.{6})([^’”])", r"\1\n\2", txt_str)
        txt_str = re.sub("([。？！\…{2}\.{6}][”’])([^，。？！])", r"\1\n\2", txt_str)
        txt_str = txt_str.strip()
        txt_str = txt_str.replace(" ", "")
        split_txt_str = txt_str.split("\n")
        all_split_len = len(split_txt_str)
        for index in range(0, all_split_len, 6):
            yield split_txt_str[index:index + 6]

    @staticmethod
    def read_sentence(files_list):
        for file in files_list:
            with open(file, "r", encoding="utf8") as fp:
                txt_read = fp.read()
                return TextProcess.cut_sentence(txt_read)


class DiaTextProcess(TextProcess):
    r"""Chinese Dialogue Text Process
        This is a module provided dialogue preprocess in Chinese Text.

        Args:
            in_features:file_path
    """
    def __init__(self, files_path, ner_model, output_path):
        super(DiaTextProcess, self).__init__(files_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model).to(Configuration.device)
        self.output_path = output_path

    @classmethod
    def modify_tag(cls, args):
        file_path = args.filepath
        file_name = args.filename
        false_role_pos = args.false_role_pos
        dialogue_pos = args.dialogue_pos
        json_path = os.path.join(file_path, file_name)
        with open(json_path, 'r', encoding="utf8") as fp:
            tag_json = json.load(fp)
        for pos in false_role_pos:
            tag_json[0]["tags"][pos] = "false_role"
        for pos in dialogue_pos:
            tag_json[0]["tags"][pos] = "dialogue"
        with open(json_path, 'w', encoding="utf8") as fp:
            json.dump(tag_json, fp, ensure_ascii=False)
        return "Success"

    @staticmethod
    def get_single_sample(sample_file):
        files_gener = os.walk(sample_file)
        for root, dirs, files in files_gener:
            for file_name in files:
                if not file_name.endswith(".json"):
                    continue
                json_path = os.path.join(root, file_name)
                with open(json_path, 'r', encoding="utf8") as fp:
                    single_sample = json.load(fp)
                for content in single_sample:
                    yield content

    @classmethod
    def split_sample(cls, sample_file):
        for content in DiaTextProcess.get_single_sample(sample_file):
            tags_list = content["tags"]
            dialogue_pos = [index for index, value in enumerate(tags_list)
                            if value == "dialogue"]
            dialogue_pos_len = len(dialogue_pos)
            if not dialogue_pos_len or dialogue_pos_len % 2:
                continue
            ## 对话人物标定模式 A -“A” B - “B” C - "C" - C, 其中“”代表对话引用， 相同字母代表相关标注。
            tmp_pos = []
            dict_dialog_pos = defaultdict(list)
            srt_dialog_pos = 0
            dia_key = "%d_%d" % (dialogue_pos[srt_dialog_pos],
                                 dialogue_pos[srt_dialog_pos + 1])
            pre_dialogue_tag = 0
            for pos, tag in enumerate(tags_list):
                if pre_dialogue_tag and tag != "dialogue":
                    continue
                if tag == "dialogue" and pre_dialogue_tag:
                    srt_dialog_pos += 2
                    srt_dialog_pos = min(srt_dialog_pos, dialogue_pos_len - 2)
                    dia_key = str(dialogue_pos[srt_dialog_pos]) + '_' \
                              + str(dialogue_pos[srt_dialog_pos + 1])
                    pre_dialogue_tag = 1 - pre_dialogue_tag
                    continue
                elif tag == "dialogue" and not pre_dialogue_tag:
                    pre_dialogue_tag = 1 - pre_dialogue_tag
                    continue
                if tag:
                    tmp_pos.append(pos)
                elif not tag and tmp_pos:
                    dict_dialog_pos[dia_key].append(tmp_pos)
                    tmp_pos = []
            if tmp_pos:
                dict_dialog_pos[dia_key].append(tmp_pos)
            for key, value in dict_dialog_pos.items():
                yield key, value, tags_list, content

    @classmethod
    def generate_train_sample(cls, sample_file, output_train):
        id_next = 0
        for key, value, tags_list, content in cls.split_sample(sample_file):
            srt, end = key.split('_')
            for ner_pos in value:
                init_tags = [''] * len(tags_list)
                init_tags[int(srt)] = "dialogue"
                init_tags[int(end)] = "dialogue"
                for pos in ner_pos:
                    init_tags[pos] = tags_list[pos]
                content["tags"] = init_tags
                with open(
                        os.path.join(output_train, "%06d_train.json" % id_next), "w", encoding="utf8") as fp:
                    json.dump([content], fp, ensure_ascii=False)
                    id_next += 1
    '''
        NER模型直接得到的是实体数字标签【1】，需要通过计算过滤出人物【1】，同时转换数字序列为字符序列【2】【3】。
    '''
    ## 【3】
    def parse_mask(self, mask_ner):
        n = len(mask_ner)
        m = len(mask_ner[0])
        for i_sent in range(n):
            for j_tokn in range(m):
                if mask_ner[i_sent][j_tokn]:
                    mask_ner[i_sent][j_tokn] = "role"
                else:
                    mask_ner[i_sent][j_tokn] = ""
        return mask_ner

    ## 【2】
    def _rep_mask_by_rule(self, txt, tar_str, mask):
        txt_len = len(txt)
        tar_str_len = len(tar_str)
        rep_tok = ["role"] * tar_str_len
        for pos in range(txt_len - tar_str_len + 1):
            if txt[pos:pos+tar_str_len] == tar_str:
                mask[pos:pos+tar_str_len] = rep_tok
        return mask

    ## 【1】
    def _generate_mask_ner(self):
        for input_txt in tqdm(DiaTextProcess.read_sentence(self.files_list)):
            inputs = self.tokenizer(input_txt, padding=True, return_tensors='pt')
            inputs = inputs.to(Configuration.device)
            output = torch.argmax(self.model(**inputs).logits, dim=-1)
            outputs = F.one_hot(output)
            outputs = outputs.to(Configuration.device)
            padder = torch.zeros(
                outputs.size()[0], outputs.size()[1],
                len(Configuration.label_list) - outputs.size()[2]
            )
            padder = padder.to(Configuration.device)
            outputs_all = torch.cat([outputs, padder], dim=-1)
            label_list = torch.Tensor([Configuration.label_list])
            label_list = label_list.to(Configuration.device)
            mask_ner = torch.matmul(
                outputs_all, label_list.T
            ).squeeze(-1).tolist()
            mask_ner = self.parse_mask(mask_ner)
            yield input_txt, mask_ner

    ##留意英文单词的分词
    def generate_tag(self):
        id_next = 0
        list_samples = []
        for input_txt, mask_ner in self._generate_mask_ner():
            dict_sent = {}
            dict_sent["text"] = ""
            dict_sent["words"] = []
            dict_sent["spaces"] = []
            dict_sent["sent_starts"] = []
            dict_sent["entities"] = []
            dict_sent["tags"] = []
            for index, txt in enumerate(input_txt):
                mask_ner[index] = mask_ner[index][1:len(txt) + 1]
                for rule in Configuration.rule_list:
                    mask_ner[index] = self._rep_mask_by_rule(
                        txt, rule, mask_ner[index]
                    )
                if len(txt) != len(mask_ner[index]):
                    continue
                words = list(txt)
                dict_sent["words"] += words
                dict_sent["text"] += " ".join(words)
                dict_sent["spaces"] += [" "] * (len(words) - 1) + [""]
                dict_sent["sent_starts"] += [False] * len(words)
                dict_sent["entities"] += ["O"] * len(words)
                dict_sent["tags"] += mask_ner[index]
            try:
                dict_sent["tags"][0] = "role"
                dict_sent["tags"][1] = "false_role"
                dict_sent["tags"][2] = "dialogue"
                dict_sent["sent_starts"][0] = True
            except IndexError:
                continue
            if id_next % 30 == 0 and id_next:
                with open(os.path.join(self.output_path, "%06d.json" % id_next), "w", encoding="utf8") as f:
                    json.dump(list_samples, f, ensure_ascii=False)
                list_samples = []
            list_samples.append(dict_sent)
            id_next += 1
