import os
import json
from tqdm import tqdm

sample_results_dir = None
ref_results_dir = None

n_samples = 32

gen_results_dir = os.path.join(sample_results_dir, f"{n_samples}")

target_dir = None

# version_and_types = [("a", "hard"), ("b", "soft"), ("c", "half"), ("d", "any"), ("a+", "hard"), ("b+", "soft"), ("c+", "half"), ("d+", "any")]
version_and_types = [("e", "hard"), ("f", "soft"), ("g", "half"), ("h", "any")]

# "hard" or "soft" or "half" or "any"

eval_type = "soft"

# models = ["llama3.2-1b", "llama3.2-3b", "llama-7b", "llama2-7b"]
# models = ["llama-7b", "llama2-7b", "llama3.2-1b", "llama3.2-3b"]
# models = ["llama3-8b"]
models = ["llama3.2-1b"]

ref_model = None
# ref_model = "llama3.1-70b"

def get_ref(fn, model, ref_model=None):
    if ref_model is None:
        return None
    fn = fn.replace(sample_results_dir, ref_results_dir)
    fn = fn.replace(str(n_samples), "1")
    fn = fn.replace(model, ref_model)
    return fn

# version = "a"
# train_type = "hard"

for version, train_type in version_and_types:

    for model in models:

        train_data_dir = os.path.join(target_dir, f"train_{version}", f"{model}-{train_type}")
        eval_data_dir = os.path.join(target_dir, f"eval_{version}", f"{model}-{eval_type}")

        os.makedirs(train_data_dir, exist_ok=True)
        os.makedirs(eval_data_dir, exist_ok=True)


        for dataset in tqdm(os.listdir(gen_results_dir)):
            
            def transfer(file_name_in, file_name_out, type, model, ref_model=None):
                assert type in ["hard", "soft", "half", "any"]
                ref_passed = 1
                
                if ref_model is not None:
                    ref_file_name_in = get_ref(file_name_in, model, ref_model)
                else:
                    ref_file_name_in = None

                with open(file_name_in, "r") as fin, open(file_name_out, "w") as fout:
                    if ref_file_name_in is not None:
                        fref = open(ref_file_name_in, "r")

                    if type in ["hard", "soft"]:
                        ref_passed = 1
                        for i, line in enumerate(fin.readlines()):
                            if i % n_samples == 0 and ref_file_name_in is not None:
                                ref_passed = json.loads(fref.readline())["passed"]
                            
                            if type == "hard":
                                data = json.loads(line)
                                data.pop("completion")
                                if not ref_passed:
                                    data["passed"] = 1
                                fout.write(json.dumps(data) + "\n")

                            elif type == "soft":

                                data = json.loads(line)
                                
                                if i % n_samples == 0:
                                    pass_ = 0    

                                if not ref_passed:
                                    data["passed"] = 1

                                pass_ += data["passed"]

                                if i % n_samples == n_samples - 1:
                                    data.pop("completion")
                                    data["passed"] = pass_ / n_samples
                                    fout.write(json.dumps(data) + "\n")
                    else:
                        if type == "half":
                            passeds = []
                            ref_passed = 1
                            for i, line in enumerate(fin.readlines()):
                                if i % n_samples == 0 and ref_file_name_in is not None:
                                    ref_passed = json.loads(fref.readline())["passed"]
                                
                                data = json.loads(line)
                                
                                if i % n_samples == 0:
                                    pass_ = 0    

                                if not ref_passed:
                                    data["passed"] = 1

                                pass_ += data["passed"]

                                if i % n_samples == n_samples - 1:
                                    passeds.append(pass_ / n_samples)
                            bound = sorted(passeds)[len(passeds) // 2]
                        else:
                            bound = 1 / n_samples
                        
                        fin.seek(0)
                        for i, line in enumerate(fin.readlines()):
                            if i % n_samples == 0 and ref_file_name_in is not None:
                                ref_passed = json.loads(fref.readline())["passed"]
                            
                            data = json.loads(line)
                            
                            if i % n_samples == 0:
                                pass_ = 0    

                            if not ref_passed:
                                data["passed"] = 1

                            pass_ += data["passed"]

                            if i % n_samples == n_samples - 1:
                                data.pop("completion")
                                data["passed"] = 1 if pass_ / n_samples >= bound else 0
                                fout.write(json.dumps(data) + "\n")
                                

                    if ref_file_name_in is not None:
                        fref.close()

            
            if dataset.endswith("train"):
                for root, dirs, files in os.walk(os.path.join(gen_results_dir, dataset)):
                    for file in files:
                        if file == f"{model}.jsonl" and all(f"{model}.jsonl_results.jsonl" != fn for fn in files):
                            transfer(os.path.join(root, file), os.path.join(train_data_dir, f"{dataset}.jsonl"), train_type, model, ref_model)
                        elif file == f"{model}.jsonl_results.jsonl":
                            transfer(os.path.join(root, file), os.path.join(train_data_dir, f"{dataset}.jsonl"), train_type, model, ref_model)

            else:
                for root, dirs, files in os.walk(os.path.join(gen_results_dir, dataset)):
                    for file in files:
                        if file == f"{model}.jsonl" and all(f"{model}.jsonl_results.jsonl" != fn for fn in files):
                            transfer(os.path.join(root, file), os.path.join(eval_data_dir, f"{dataset}.jsonl"), eval_type, model, ref_model)
                        elif file == f"{model}.jsonl_results.jsonl":
                            transfer(os.path.join(root, file), os.path.join(eval_data_dir, f"{dataset}.jsonl"), eval_type, model, ref_model)
