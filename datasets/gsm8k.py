
import os, json
from torch.utils.data import Dataset
from .general import gen_collate_fn_v1 as collate_fn_v1
from .general import gen_post_process_func_v1, ppl_post_process_func_v1
from .general import judge_gen
from utils.jsonl import stream_jsonl


class GSM8kGenDataset(Dataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/gsm8k/test.jsonl",
    ):
        self.data_path = data_path
        self.datas = []

        assert os.path.isfile(data_path) and data_path.endswith("jsonl")

        for i, line in enumerate(stream_jsonl(data_path)):
            data = {
                "task_id": i,
                "prompt": line["question"],
                "answers": [line["answer"].split("####")[1].strip()],
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


class GSM8kTrainGenDataset(GSM8kGenDataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/gsm8k/train.jsonl",
    ):
        super().__init__(data_path)


class ArithmeticTrainGenDataset(GSM8kGenDataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/arithmetic/train.jsonl",
    ):
        super().__init__(data_path)


class ArithmeticInterGenDataset(GSM8kGenDataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/arithmetic/interpolate.jsonl",
    ):
        super().__init__(data_path)


class ApeTrainGenDataset(GSM8kGenDataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/ape/train.jsonl",
    ):
        super().__init__(data_path)

class ApeValidationGenDataset(GSM8kGenDataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/ape/validation.jsonl",
    ):
        super().__init__(data_path)

class ApeTestGenDataset(GSM8kGenDataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/ocdata/ape/test.jsonl",
    ):
        super().__init__(data_path)


gsm8k_gen_conf = {
    "dataset": GSM8kGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "version": "v1",
    "type": "generation",
}

gsm8k_train_gen_conf = {
    "dataset": GSM8kTrainGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "version": "v1",
    "type": "generation",
}

gsm8k_ppl_conf = {
    "dataset": GSM8kGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": ppl_post_process_func_v1,
    "version": "v1",
    "type": "ppl",
}

arithmetic_train_gen_conf = {
    "dataset": ArithmeticTrainGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "version": "v1",
    "type": "generation",
}

arithmetic_inter_gen_conf = {
    "dataset": ArithmeticInterGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "version": "v1",
    "type": "generation",
}

arithmetic_inter_ppl_conf = {
    "dataset": ArithmeticInterGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": ppl_post_process_func_v1,
    "version": "v1",
    "type": "ppl",
}

ape_train_gen_conf = {
    "dataset": ApeTrainGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "type": "generation",
}

ape_validation_gen_conf = {
    "dataset": ApeValidationGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "type": "generation",
}

ape_test_gen_conf = {
    "dataset": ApeTestGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": gen_post_process_func_v1,
    "type": "generation",
}

ape_train_ppl_conf = {
    "dataset": ApeTrainGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": ppl_post_process_func_v1,
    "type": "ppl",
}

ape_validation_ppl_conf = {
    "dataset": ApeValidationGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": ppl_post_process_func_v1,
    "type": "ppl",
}

ape_test_ppl_conf = {
    "dataset": ApeTestGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": ppl_post_process_func_v1,
    "type": "ppl",
}