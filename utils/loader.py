from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
import os

def load_model_and_tokenizer_gen(
    model_repoid_or_path, 
    tp=1, 
    max_tokens=200, 
    temperature=0., 
    **kwargs,
):
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')

    pipe = LLM(model=model_repoid_or_path, tensor_parallel_size=tp)
    
    tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    return pipe, tokenizer, sampling_params

def load_model_and_tokenizer_ppl(
    model_repoid_or_path, 
    tp=1,
    **kwargs,
):
    
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')

    print(f"{kwargs} do not have any effect in load_model_and_tokenizer_ppl")

    pipe = LLM(model=model_repoid_or_path, tensor_parallel_size=tp)
    
    tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path, trust_remote_code=True)

    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    return pipe, tokenizer, sampling_params


def load_model_and_tokenizer_pik(
    model_repoid_or_path, 
    tp=1,
    **kwargs,
):
    
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')

    print(f"{kwargs} do not have any effect in load_model_and_tokenizer_ppl")

    pipe = LLM(model=model_repoid_or_path, tensor_parallel_size=tp, task="embedding")
    
    tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path, trust_remote_code=True)

    if os.path.exists(os.path.join(model_repoid_or_path, "wik.pt")):
        wik = torch.load(os.path.join(model_repoid_or_path, "wik.pt"), map_location="cpu")["weight"]
    else:
        wik = None

    return pipe, tokenizer, wik


load_fns = {
    "generation": load_model_and_tokenizer_gen,
    "ppl": load_model_and_tokenizer_ppl,
    "pik": load_model_and_tokenizer_pik,
}
