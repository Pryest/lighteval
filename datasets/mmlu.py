import os, json
import pandas as pd
from torch.utils.data import Dataset
from .general import ppl_post_process_func_v1 as post_process_func_v1
from .general import judge_ppl

mmlu_all_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

class MMLUpplDataset(Dataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/mmlu/dev/",
    ):
        self.data_path = data_path
        self.datas = []

        id_cnt = 0

        for _name in sorted(mmlu_all_sets):
            file_path = os.path.join(data_path, f"{_name}_dev.csv")
            assert os.path.isfile(file_path) and file_path.endswith("csv")

            df = pd.read_csv(file_path, sep=",")

            for i in range(len(df)):
                data = {
                    "task_id": id_cnt,
                    "_name": _name,
                    "prompt": df.iloc[i, 0],
                    "A": df.iloc[i, 1],
                    "B": df.iloc[i, 2],
                    "C": df.iloc[i, 3],
                    "D": df.iloc[i, 4],
                    "answer": df.iloc[i, 5],
                }
                self.datas.append(data)
                id_cnt += 1


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        return data

    @staticmethod
    def judge(ref_data, ppl_data):
        return judge_ppl(ref_data, ppl_data)
    

def collate_fn_v1(batch):
    ids = []
    prompts = []
    for x in batch:
        ids.append(x["task_id"])
        _hint = f"The following are multiple choice questions (with answers) about {x['_name'].replace('_', ' ')}.\n\n"
        prompts.append("{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:".format(input=_hint + x["prompt"], A=x["A"], B=x["B"], C=x["C"], D=x["D"]))
    return {"ids": ids, "prompts": prompts,}


mmlu_ppl_conf = {
    "dataset": MMLUpplDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": post_process_func_v1,
    "version": "v1",
    "type": "ppl",
}

