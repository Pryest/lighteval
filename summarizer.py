import re
import os


pattern = r"judge results: [\d]+/[\d]+ = ([\d\.]+)"

results = []

for root, dirs, files in os.walk("../logs"):
    for file in files:
        if file.endswith(".log"):
            model_task = file.replace(".log", "")
            model = model_task.split("_")[0]
            task = "_".join(model_task.split("_")[1:])
            with open(os.path.join(root, file), "r") as f:
                try:
                    text = f.read()
                except:
                    print(root, file)
                res = re.findall(pattern, text)
                if res:
                    results.append((model, task, float(res[0])))

results.sort(key=lambda x: (x[1], x[2]), reverse=True)

for res in results:
    print(*res)
