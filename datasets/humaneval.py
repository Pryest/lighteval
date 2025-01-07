import os, json
from torch.utils.data import Dataset
from .general import ppl_post_process_func_v1

class HumanevalGenDataset(Dataset):
    def __init__(self, 
        data_path=f"{os.environ['datadir']}/hedata/HumanEval.jsonl",
        required_fields=["task_id", "prompt"],
    ):
        self.data_path = data_path
        self.datas = []

        assert os.path.isfile(data_path) and data_path.endswith("jsonl")

        with open(data_path, "r") as f:
            for line in list(f.readlines()):
                data = json.loads(line)
                data = {k: data[k].lstrip("\n") for k in required_fields}
                self.datas.append(data)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        return data

def collate_fn_v1(batch):
    ids = []
    prompts = []
    for x in batch:
        ids.append(x["task_id"])
        prompts.append("Complete the following python code:\n\n" + x["prompt"])
    return {"ids": ids, "prompts": prompts,}


# def collate_fn_v2(batch):
#     ids = []
#     prompts = []
#     for x in batch:
#         ids.append(x["task_id"])
#         prompts.append("Complete the following python code:\n\n" + x["prompt"] + "    # your code here\n")
#     return {"ids": ids, "prompts": prompts,}


def post_process_func_v1(
    ids,
    prompts,
    outputs,
    n=1,
):
    save_datas = []
    for gid, prompt, output in zip(ids, prompts, outputs):
        for j in range(n):
            split_text = output.outputs[j].text.split("\n\n\n")
            for i in range(len(split_text)):
                if len(split_text[i].strip()) > 0 and not split_text[i].strip().startswith("#"):
                    completion = "\n\n\n".join(split_text[:i+1])
                    break
            else:
                completion = ""
            
            save_data = dict(
                task_id=gid, 
                prompt=prompt, 
                completion=completion,
                original_response=output.outputs[j].text
            )

            save_datas.append(save_data)
    return save_datas

humaneval_gen_conf = {
    "dataset": HumanevalGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": post_process_func_v1,
    "version": "v1",
    "type": "generation",
}

# humaneval_gen_conf = {
#     "dataset": HumanevalGenDataset,
#     "collate_fn": collate_fn_v2,
#     "post_process_func": post_process_func_v1,
#     "version": "v2",
#     "type": "generation",
# }

humaneval_ppl_conf = {
    "dataset": HumanevalGenDataset,
    "collate_fn": collate_fn_v1,
    "post_process_func": ppl_post_process_func_v1,
    "version": "v1",
    "type": "ppl",
}