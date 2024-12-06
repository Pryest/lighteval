from .humaneval import humaneval_gen_conf
from .nq import nq_gen_conf
from .triviaqa import triviaqa_gen_conf
from .mmlu import mmlu_ppl_conf

infer_confs = {
    "humaneval": humaneval_gen_conf,
    "nq": nq_gen_conf,
    "triviaqa": triviaqa_gen_conf,
    "mmlu": mmlu_ppl_conf,
}