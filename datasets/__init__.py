from .humaneval import humaneval_gen_conf, humaneval_ppl_conf
from .nq import nq_gen_conf, nq_ppl_conf
from .triviaqa import triviaqa_gen_conf, triviaqa_ppl_conf
from .mmlu import mmlu_ppl_conf
from .gsm8k import gsm8k_gen_conf, gsm8k_train_gen_conf, gsm8k_ppl_conf
from .triviaqa_train import triviaqa_train_gen_conf
from .nq_train import nq_train_gen_conf

infer_confs = {
    "humaneval": humaneval_gen_conf,
    "nq": nq_gen_conf,
    "triviaqa": triviaqa_gen_conf,
    "mmlu": mmlu_ppl_conf,
    "gsm8k": gsm8k_gen_conf,
    "triviaqa_train": triviaqa_train_gen_conf,
    "gsm8k_train": gsm8k_train_gen_conf,
    "nq_train": nq_train_gen_conf,
    "nq_ppl": nq_ppl_conf,
    "triviaqa_ppl": triviaqa_ppl_conf,
    "humaneval_ppl": humaneval_ppl_conf,
    "gsm8k_ppl": gsm8k_ppl_conf,
}