from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from .jsonl import stream_jsonl, write_jsonl
import os
import shutil

def infer_with_conf(
    llm,
    sampling_params,
    infer_conf, 
    save_path, 
    infer_only,
    judge_only,
    try_resume,
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

    if not judge_only and not os.path.exists(save_path):

        if try_resume:
            resume_folder = save_path.with_suffix('.backup')
            os.makedirs(resume_folder, exist_ok=True)
        else:
            resume_folder = None

        save_datas = []
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        dl = tqdm(dl) if enable_tqdm else dl

        idx = -1

        for idx, kwargs in enumerate(dl):
            if try_resume:
                if os.path.exists(resume_folder / f"{idx}.jsonl"):
                    continue
                else:
                    save_datas = []

            outputs = llm.generate(kwargs["prompts"], sampling_params)
            kwargs["outputs"] = outputs

            if post_process_func:
                save_datas.extend(post_process_func(**kwargs, n=n))
            else:
                save_datas.extend(kwargs)

            if try_resume:
                write_jsonl(resume_folder / f"{idx}.jsonl", save_datas)
        
        if try_resume:
            save_datas = []
            for i in range(idx + 1):
                save_datas.extend(stream_jsonl((resume_folder / f"{i}.jsonl").as_posix()))

        write_jsonl(save_path, save_datas)

        if try_resume:
            shutil.rmtree(resume_folder)

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
    try_resume,
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

        if infer_only and os.path.exists(save_path.parent / "0.pt"):
            print(f"{save_path.parent}/0.pt exists, skip")
            return
        
        if not infer_only and os.path.exists(save_path):
            print(f"{save_path} exists, skip")
            return

        if try_resume:
            resume_folder = save_path.with_suffix('.backup')
            os.makedirs(resume_folder, exist_ok=True)

        save_datas = []
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        dl = tqdm(dl)

        idx = -1
        for idx, kwargs in enumerate(dl):
            if try_resume:
                if os.path.exists(resume_folder / f"{idx}.jsonl") or os.path.exists(resume_folder / f"{idx}.pt"):
                    continue
                else:
                    save_datas = []
            
            outputs = llm.encode(kwargs["prompts"])
            for id, prompt, output in zip(kwargs["ids"], kwargs["prompts"], outputs):
                if wik is None:
                    logits = torch.tensor(output.outputs.embedding, dtype=torch.float32, device="cpu")
                else:
                    logits = torch.tensor(output.outputs.embedding, dtype=wik.dtype, device=wik.device)

                if infer_only:
                    save_datas.append({"task_id": id, "logits": logits})
                    if try_resume:
                        torch.save(save_datas, resume_folder / f"{idx}.pt")
                else:
                    pred = torch.nn.functional.softmax(torch.matmul(wik, logits), dim=-1)[1].item()
                    save_datas.append({"task_id": id, "prompt": prompt, "P(IK)": pred})
                    if try_resume:
                        write_jsonl(resume_folder / f"{idx}.jsonl", save_datas)

        if infer_only:
            if try_resume:
                save_datas = []
                for i in range(idx + 1):
                    save_datas.extend(torch.load(resume_folder / f"{i}.pt"))
            torch.save(save_datas, save_path.parent / "0.pt")
        else:
            if try_resume:
                save_datas = []
                for i in range(idx + 1):
                    save_datas.extend(stream_jsonl((resume_folder / f"{i}.jsonl").as_posix()))
            write_jsonl(save_path, save_datas)

        if try_resume:
            shutil.rmtree(resume_folder)
