from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from .jsonl import stream_jsonl, write_jsonl

def infer_with_conf(
    llm,
    sampling_params,
    infer_conf, 
    save_path, 
    infer_only,
    judge_only,
    n=1,
    seed=1001,
    enable_tqdm=True
):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = infer_conf["dataset"]
    batch_size = infer_conf["batch_size"]
    collate_fn = infer_conf["collate_fn"]
    post_process_func = infer_conf.get("post_process_func", None)
    version = infer_conf.get("version", None)

    if judge_only and infer_only:
        raise ValueError("judge_only and infer_only cannot be both True")

    ds = dataset()

    if not judge_only:
        save_datas = []
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        dl = tqdm(dl) if enable_tqdm else dl

        for kwargs in dl:
            outputs = llm.generate(kwargs["prompts"], sampling_params)
            kwargs["outputs"] = outputs

            if post_process_func:
                save_datas.extend(post_process_func(**kwargs, n=n))
            else:
                save_datas.extend(kwargs)
        
        write_jsonl(save_path, save_datas)

    if not infer_only and hasattr(dataset, "judge"):
        cnt, tot = 0, 0
        save_datas = []

        for i, save_data in enumerate(stream_jsonl(str(save_path))):
            judge_results = dataset.judge(ds[i//n], save_data)
            save_data.update(judge_results)
            save_datas.append(save_data)
            cnt += judge_results["passed"]
            tot += 1

        write_jsonl(save_path, save_datas)
        print(f"judge results: {cnt}/{tot} = {cnt/tot:.4f}")


def pred_pik(
    llm,
    wik,
    infer_conf, 
    save_path, 
    infer_only,
    seed=1001,
    **kwargs,
):
    print(f"kwargs not working in pred_pik: {kwargs}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad():

        dataset = infer_conf["dataset"]
        batch_size = infer_conf["batch_size"]
        collate_fn = infer_conf["collate_fn"]
        version = infer_conf.get("version", None)

        ds = dataset()

        save_datas = []
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        dl = tqdm(dl)
        for kwargs in dl:
            outputs = llm.encode(kwargs["prompts"])
            for id, prompt, output in zip(kwargs["ids"], kwargs["prompts"], outputs):
                logits = torch.tensor(output.outputs.embedding, dtype=wik.dtype, device=wik.device)
                if infer_only:
                    save_datas.append({"task_id": id, "logits": logits})
                else:
                    pred = torch.nn.functional.softmax(torch.matmul(wik, logits), dim=-1)[1].item()
                    save_datas.append({"task_id": id, "prompt": prompt, "P(IK)": pred})

        if infer_only:
            torch.save(save_datas, save_path.parent / "0.pt")
        else:
            write_jsonl(save_path, save_datas)
