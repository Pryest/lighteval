from argparse import ArgumentParser
import time
from datetime import datetime
from pathlib import Path
import os

from utils.launcher import run, local_models, get_tp

def system_1(
    model_name, 
    dataset_name,
    save_dir,
    batch_size, 
    infer_only,
    judge_only,
):
    from datasets import infer_confs
    from utils.infer import infer_with_conf
    from utils.loader import load_fns
    
    model_path = local_models[model_name]
    tp = get_tp(model_name)
    
    infer_conf = infer_confs[dataset_name]
    infer_conf["batch_size"] = batch_size

    version = infer_conf.get("version", "v0")
    save_dir = Path(save_dir) / dataset_name / version
    os.makedirs(save_dir, exist_ok=True)
    save_file = save_dir / (model_name + ".jsonl")

    load_fn = load_fns[infer_conf["type"]]

    llm, tokenizer, sampling_params = load_fn(model_path, tp)
    infer_with_conf(llm, sampling_params, infer_conf, save_file, infer_only, judge_only)


def system_2(
    launch_time,
    model_name, 
    dataset_name, 
    log_dir, 
    batch_size,
    infer_only,
    judge_only,
):
    entry_file = __file__
    log_dir = Path(log_dir) / launch_time
    os.makedirs(log_dir, exist_ok=True)
    run(model_name, dataset_name, entry_file, log_dir, batch_size, infer_only, judge_only)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="all")
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--log_dir", type=str, default=f"/cpfs01/shared/public/{os.environ['USER']}/1205/logs")
    parser.add_argument("--save_dir", type=str, default=f"/cpfs01/shared/public/{os.environ['USER']}/1205/results")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--infer_only", action="store_true")
    parser.add_argument("--judge_only", action="store_true")
    parser.add_argument("--auto_launch", action="store_true")

    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(f"unknown_args: {unknown_args}")

    if not args.auto_launch:

        launch_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%H:%M:%S')

        cmd_dir = Path(__file__).parent / "cmds" / launch_time

        os.makedirs(cmd_dir, exist_ok=True)

        with open(cmd_dir / "run.sh", "w") as f:
            f.write("python run.py")
            f.write(f" \\ \n --model_name {args.model_name}")
            f.write(f" \\ \n --dataset_name {args.dataset_name}")
            f.write(f" \\ \n --log_dir {args.log_dir}")
            f.write(f" \\ \n --save_dir {args.save_dir}")
            f.write(f" \\ \n --batch_size {args.batch_size}")
            if args.infer_only:
                f.write(" \\ \n --infer_only")
            if args.judge_only:
                f.write(" \\ \n --judge_only")

    if args.auto_launch:
        system_1(args.model_name, args.dataset_name, args.save_dir, args.batch_size, args.infer_only, args.judge_only)
    else:
        system_2(launch_time, args.model_name, args.dataset_name, args.log_dir, args.batch_size, args.infer_only, args.judge_only)