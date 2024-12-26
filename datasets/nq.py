import os, json
import pandas as pd
from torch.utils.data import Dataset
from .general import gen_collate_fn_v1 as collate_fn_v1
from .general import gen_post_process_func_v1 as post_process_func_v1
from .general import judge_gen


class NQGenDataset(Dataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/nq/nq-test.qa.csv",
    ):
        self.data_path = data_path
        self.datas = []

        assert os.path.isfile(data_path) and data_path.endswith("csv")

        df = pd.read_csv(data_path, sep="\t")

        for i in range(len(df)):
            data = {
                "task_id": i,
                "prompt": df.iloc[i, 0],
                "answers": eval(df.iloc[i, 1]),
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


nq_gen_conf = {
    "dataset": NQGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": post_process_func_v1,
    "version": "v1",
    "type": "generation",
}