def gen_collate_fn_v1(batch):
    ids = []
    prompts = []
    for x in batch:
        ids.append(x["task_id"])
        prompts.append("Question: {question}?\nAnswer: ".format(question=x["prompt"].rstrip("?")))
    return {"ids": ids, "prompts": prompts,}

def gen_post_process_func_v1(
    ids,
    prompts,
    outputs,
):
    save_datas = []
    for gid, prompt, output in zip(ids, prompts, outputs):
        split_text = output.outputs[0].text.split("\n")
        for i in range(len(split_text)):
            if len(split_text[i].strip()) > 0 :
                completion = split_text[i].strip()
                break          
        else:
            completion = ""
        
        save_data = dict(
            task_id=gid, 
            prompt=prompt, 
            completion=completion,
            original_response=output.outputs[0].text
        )

        save_datas.append(save_data)
    return save_datas

def ppl_post_process_func_v1(
    ids,
    prompts,
    outputs,
):
    save_datas = []
    for gid, prompt, output in zip(ids, prompts, outputs):
        logprobs = sorted(list(output.outputs[0].logprobs[0].items()), key=lambda x: x[1].rank)

        preds = []
        for token_id, logprobinfo in logprobs:
            logprob = logprobinfo.logprob
            token = logprobinfo.decoded_token
            preds.append((token, logprob))
        
        save_data = dict(
            task_id=gid, 
            prompt=prompt, 
            preds=preds,
        )

        save_datas.append(save_data)

    return save_datas


def judge_gen(ref_data, gen_data):
    answers = ref_data.get("answers", [ref_data.get("answer", "NOTEVENACHANCE")])
    return {
        "passed": any([a.lower() in gen_data["completion"].lower() for a in answers])
    }

def judge_ppl(ref_data, ppl_data):
    answer = ref_data.get("answer", "NOTEVENACHANCE")
    preds = ppl_data["preds"]
    return {
        "passed": answer.strip() == preds[0][0].strip(),
        "answer": answer,
    }