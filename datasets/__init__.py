from .humaneval import humaneval_gen_conf, humaneval_ppl_conf
from .nq import nq_gen_conf, nq_ppl_conf
from .triviaqa import triviaqa_gen_conf, triviaqa_ppl_conf
from .mmlu import mmlu_ppl_conf
from .gsm8k import gsm8k_gen_conf, gsm8k_train_gen_conf, gsm8k_ppl_conf, arithmetic_train_gen_conf, arithmetic_inter_gen_conf, arithmetic_inter_ppl_conf, ape_train_gen_conf, ape_validation_gen_conf, ape_test_gen_conf, ape_train_ppl_conf, ape_validation_ppl_conf, ape_test_ppl_conf
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
    "arithmetic_train": arithmetic_train_gen_conf,
    "arithmetic_inter": arithmetic_inter_gen_conf,
    "arithmetic_inter_ppl": arithmetic_inter_ppl_conf,
    "ape_train": ape_train_gen_conf,
    "ape_validation": ape_validation_gen_conf,
    "ape_test": ape_test_gen_conf,
    "ape_train_ppl": ape_train_ppl_conf,
    "ape_validation_ppl": ape_validation_ppl_conf,
    "ape_test_ppl": ape_test_ppl_conf,
}