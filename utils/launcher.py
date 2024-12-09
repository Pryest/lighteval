from pathlib import Path
import subprocess
import os
import re
import math

local_datasets = ["triviaqa", "nq"]

local_models = {
    # "llama-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",
    # "llama-13b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba",
    # "llama-30b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--huggyllama--llama-30b/snapshots/2b1edcdb3c7ced7bce6c1aa75c94545777c3118b",
    # "llama-65b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--huggyllama--llama-65b/snapshots/49707c5313d34d1c5a846e29cf2a2a650c22c8ee",

    # "llama2-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
    # "llama2-13b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55",
    # "llama2-70b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-2-70b-hf/snapshots/3aba440b59558f995867ba6e1f58f21d0336b5bb",

    # "llama3-8b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b",
    # "llama3-70b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338",

    # "llama3.1-8b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb",
    "llama3.1-70b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/7740ff69081bd553f4879f71eebcc2d6df2fbcb3",
    
    # "llama3.2-1b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-3.2-1B/snapshots/5d853ed7d16ac794afa8f5c9c7f59f4e9c950954",
    # "llama3.2-3b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-3.2-3B/snapshots/5cc0ffe09ee49f7be6ca7c794ee6bd7245e84e60",

    # "mistral-v0.1-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
    # "mistral-v0.2-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--mistral-community--Mistral-7B-v0.2/snapshots/2c3e624962b1a3f3fbf52e15969565caa7bc064a",
    # "mistral-v0.3-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--mistralai--Mistral-7B-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8",

    # "mixtral-v0.1-8x7-56b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/58301445dc1378584211722b7ebf8743ec4e192b",
    # "mixtral-v0.1-8x22-176b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--mistralai--Mixtral-8x22B-v0.1/snapshots/b03e260818710044a2f088d88fab12bb220884fb",

    # "gemma-2b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--google--gemma-2b/snapshots/2ac59a5d7bf4e1425010f0d457dde7d146658953",
    # "gemma-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--google--gemma-7b/snapshots/a0eac5b80dba224e6ed79d306df50b1e92c2125d",
    # "gemma2-2b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--google--gemma-2-2b/snapshots/0738188b3055bc98daf0fe7211f0091357e5b979",
    # "gemma2-9b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--google--gemma-2-9b/snapshots/7305b337e801768dc5c40319c421052afac25b77",
    # "gemma2-27b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--google--gemma-2-27b/snapshots/4333076ac61b76ec1996a01cc962d7d7dcdb2cc5",
    
    # "qwen-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen-7B/snapshots/ef3c5c9c57b252f3149c1408daf4d649ec8b6c85",
    # "qwen-14b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen-14B/snapshots/5ea0d35feedf5bb275800b986244d4703dbab6ea",
    # "qwen-72b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen-72B/snapshots/6dae9c86d4f42c3be60ffb525f784ee1991605cb",

    # "qwen1.5-0.5b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-0.5B/snapshots/8f445e3628f3500ee69f24e1303c9f10f5342a39",
    # "qwen1.5-1.8b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-1.8B/snapshots/7846de7ed421727b318d6605a0bfab659da2c067",
    # "qwen1.5-4b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-4B/snapshots/294dbdee5dacecc52c9cc6ba2dba4084addc7b2c",
    # "qwen1.5-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-7B/snapshots/831096e3a59a0789a541415da25ef195ceb802fe",
    # "qwen1.5-14b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-14B/snapshots/39b74a78357df4d2296e838d87565967d663a67a",
    # "qwen1.5-32b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-32B/snapshots/cefef80dc06a65f89d1d71d0adbc56d335ca2490",
    # "qwen1.5-72b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen1.5-72B/snapshots/93bac0d1ae83d50c43b1793e2d74a00dc43a4c36",

    # "qwen2-0.5b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2-0.5B/snapshots/3324425e67bd0278078051a8b08d2be9aa810247",
    # "qwen2-1.5b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2-1.5B/snapshots/537f5abdac43d8045da79757d0a3e481a3d0b699",
    # "qwen2-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e",
    # "qwen2-57b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2-57B-A14B/snapshots/973e466c39ba76372a2ae464dbca0af3f5a5a2a9",
    # "qwen2-72b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2-72B/snapshots/87993795c78576318087f70b43fbf530eb7789e7",

    # "qwen2.5-0.5b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-0.5B/snapshots/7b382a7278a945cf842898802dd4d862d54c513f",
    # "qwen2.5-1.5b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323",
    # "qwen2.5-3b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    # "qwen2.5-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-7B/snapshots/09a0bac5707b43ec44508eab308b0846320c1ed4",
    # "qwen2.5-32b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-32B/snapshots/1818d35814b8319459f4bd55ed1ac8709630f003",
    # "qwen2.5-72b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-72B/snapshots/587cc4061cf6a7cc0d429d05c109447e5cf063af",

    # "qwen2.5-coder-1.5b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-Coder-1.5B/snapshots/ad88ed4e19c97bb09126419e8a99213692e28eb2",
    # "qwen2.5-coder-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-Coder-7B/snapshots/8e82b34bef0b4c1ae4d310b646f3ee9790708621",
    # "qwen2.5-math-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e",
    # "qwen2.5-math-72b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-Math-72B/snapshots/b3fd85a64ec2691b67c07ef45de24dcb04e25e5c",

    # "internlm-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm-7b/snapshots/712d9412503729a4a4e9e605003400a52c59865d",

    # "internlm2-1.8b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2-1_8b/snapshots/35f91cdd711158997d4832000c7697985a8c968a",
    # "internlm2-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2-base-7b/snapshots/438e4c8a828b216602eb29eefd2640f0fa6ace41",
    # "internlm2-20b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2-base-20b/snapshots/24e1e91e50024a670198bb4dc7bc847cad1deba2",

    # "internlm2.5-1.8b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-1_8b/snapshots/acd378747fb43ddfe045c98304a07b71e53f4ef4",
    # "internlm2.5-7b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-7b/snapshots/7bee5c5d7c4591d7f081f5976610195fdd1f1e35",
    # "internlm2.5-20b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-20b/snapshots/f2b8f8ad38c73b21cfa8dbc57c42bfc5fc720915",

    # "deepseek-coder-1.3b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--deepseek-ai--deepseek-coder-1.3b-base/snapshots/c919139c3a9b4070729c8b2cca4847ab29ca8d94",
    # "deepSeek-coder-v2-236b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--deepseek-ai--DeepSeek-Coder-V2-Base/snapshots/0e809b58ce4354bf3e37f882b7de727241a1afcb",
    # "deepSeek-coder-v2-16b":"/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Base/snapshots/e5e79b92c85f8f9182ec006b575483227201fd5e",
    
}

ali_h_script_format = """\
dlc create job \
--kind PyTorchJob \
--config /cpfs01/shared/public/$USER/dlc.cfg \
--name vllm_test --priority 4 \
--data_sources="" \
--worker_count 1 \
--worker_cpu {worker_cpu} \
--worker_gpu {worker_gpu} \
--worker_memory "{worker_memory}Gi" \
--worker_image=pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/paieflops/jiaopenglong:jpl-py310-torch2-1-flash2-2-1-cu118-accl \
--workspace_id=ws1os366fx8vi304 \
--worker_shared_memory "100Gi"  \
--command="bash -c '\
export USER=$USER && \
source /cpfs01/shared/public/$USER/anaconda3/bin/activate /cpfs01/shared/public/$USER/anaconda3/envs/1205/ && \
cd {work_dir} && export VLLM_USE_MODELSCOPE=False && \
python {entry_file} \
--model_name={model_name} \
--dataset_name={dataset_name} \
--batch_size={batch_size} \
{infer_only}{judge_only}--auto_launch \
2>&1 | tee {log_file} \
'"\
"""

# wsu38o3r5miz6dof     ws1os366fx8vi304


def get_tp(model_name):
    model_size = float(re.findall("([\d\.]*)b", model_name)[0])
    model_size = int(math.ceil(model_size))
    if model_size <= 9:
        return 1
    if model_size <= 14:
        return 2
    if model_size <= 32:
        return 4    
    return 8

def run(
    model_name, 
    dataset_name, 
    entry_file, 
    log_dir, 
    batch_size, 
    infer_only, 
    judge_only
):
    if model_name == "all":
        for model_name, model_path in local_models.items():
            run(model_name, dataset_name, entry_file, log_dir, batch_size, infer_only, judge_only)
    elif dataset_name == "all":
        for dataset_name in local_datasets:
            run(model_name, dataset_name, entry_file, log_dir, batch_size, infer_only, judge_only)
    else:
        work_dir = os.getcwd()
        log_file = Path(log_dir) / (model_name + "_" + dataset_name + ".log")
        tp = get_tp(model_name)
        worker_cpu = 8 * tp
        # worker_cpu = 12 * tp
        worker_gpu = tp
        worker_memory = 100 * tp
        ali_h_script = ali_h_script_format.format(
            worker_cpu = worker_cpu,
            worker_gpu = worker_gpu,
            worker_memory = worker_memory,
            entry_file = entry_file,
            model_name = model_name,
            dataset_name = dataset_name,
            log_file = log_file,
            work_dir = work_dir,
            batch_size = batch_size,
            infer_only = "--infer_only \\ \n" if infer_only else "",
            judge_only = "--judge_only \\ \n" if judge_only else "",
        )

        ret = subprocess.run(ali_h_script, shell=True)
