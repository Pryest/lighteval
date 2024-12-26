import os, json
import pandas as pd
from torch.utils.data import Dataset
from .general import gen_collate_fn_v1 as collate_fn_v1
from .general import gen_post_process_func_v1 as post_process_func_v1
from .general import judge_gen
from utils.jsonl import stream_jsonl


class TriviaQATrainGenDataset(Dataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/triviaqa/triviaqa-train.jsonl",
    ):
        self.data_path = data_path
        self.datas = []

        assert os.path.isfile(data_path) and data_path.endswith("jsonl")
        for i, line in enumerate(stream_jsonl(data_path)):
            data = {
                "task_id": i,
                "prompt": line["question"],
                "answers": [line["answer"].strip()],
            }
            self.datas.append(data)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        return data
    
    @staticmethod
    def judge(ref_data, gen_data):
        return judge_gen(ref_data, gen_data)


triviaqa_train_gen_conf = {
    "dataset": TriviaQATrainGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": post_process_func_v1,
    "version": "v1",
    "type": "generation",
}