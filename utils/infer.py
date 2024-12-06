from tqdm import tqdm
from torch.utils.data import DataLoader
from .jsonl import stream_jsonl, write_jsonl

def infer_with_conf(
    llm,
    sampling_params,
    infer_conf, 
    save_path, 
    infer_only,
    judge_only,
    enable_tqdm=True
):
    dataset = infer_conf["dataset"]
    batch_size = infer_conf["batch_size"]
    collate_fn = infer_conf["collate_fn"]
    post_process_func = infer_conf.get("post_process_func", None)
    version = infer_conf.get("version", None)

    ds = dataset()

    if not judge_only:
        save_datas = []
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        dl = tqdm(dl) if enable_tqdm else dl

        for kwargs in dl:
            outputs = llm.generate(kwargs["prompts"], sampling_params)
            kwargs["outputs"] = outputs

            if post_process_func:
                save_datas.extend(post_process_func(**kwargs))
            else:
                save_datas.extend(kwargs)
        
        write_jsonl(save_path, save_datas)

    if not infer_only:
        cnt, tot = 0, 0
        save_datas = []
        for data, save_data in zip(ds, stream_jsonl(str(save_path))):
            judge_results = dataset.judge(data, save_data)
            save_data.update(judge_results)
            save_datas.append(save_data)
            cnt += judge_results["passed"]
            tot += 1
        write_jsonl(save_path, save_datas)
        print(f"judge results: {cnt}/{tot}= {cnt/tot:.4f}")