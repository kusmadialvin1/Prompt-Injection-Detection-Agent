from transformers import AutoTokenizer
import os
import torch
import json
from PIGuard import PIGuard
from util import set_seed, get_logger
from params import parse_args

def mean_defined(values):
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None

def acc_compute(model, target_set, target_class="benign", name="chat"):
    bad_sample = []
    logits_list = []
    save_dict = []

    if not target_set:
        print(f"WARNING: {name} set is empty; accuracy is undefined and will be skipped.")
        return None

    if target_class == "benign":
        target_label = 0
    else:
        target_label = 1
    
    with torch.no_grad():
        for idx, sample in enumerate(target_set):
            class_logits = model.classify(sample)
            pred = class_logits.argmax().item()
            if pred != target_label:
                bad_sample.append(sample)
                logits_list.append(class_logits.cpu())
                save_dict.append({"prompt": sample, "logits": class_logits.cpu().squeeze().tolist()})
                # print(f"Prompt: {sample}\nLogits: {class_logits.cpu()}")

            del class_logits
            torch.cuda.empty_cache()
        
    acc = 1 - len(save_dict)/len(target_set)
    print(f"{name} set accuracy: {acc}")
    return acc


def pint_evaluate(model, dataset_root="datasets"):
    with open(os.path.join(dataset_root, "PINT.json"), "r") as f:
        valid_dataset = json.load(f)

    benign_set, injection_set = [], []
    chat_set, documents_set, hard_negatives_set, public_prompt_injection_set, internal_prompt_injection_set, jailbreak_set = [], [], [], [], [], []

    for sample in valid_dataset:
        if sample["label"] == False:
            if sample["category"] == "chat":
                chat_set.append(sample["text"])

            elif sample["category"] == "documents":
                documents_set.append(sample["text"])

            elif sample["category"] == "hard_negatives":
                hard_negatives_set.append(sample["text"])
            
            else:
                ValueError("Wrong Key")

            benign_set.append(sample["text"])

        elif sample["label"] == True:
            if sample["category"] == "public_prompt_injection":
                public_prompt_injection_set.append(sample["text"])

            elif sample["category"] == "internal_prompt_injection":
                internal_prompt_injection_set.append(sample["text"])

            elif sample["category"] == "jailbreak":
                jailbreak_set.append(sample["text"])

            else:
                ValueError("Wrong Key")

            injection_set.append(sample["text"])

    chat_acc = acc_compute(model, chat_set, target_class="benign", name="chat")
    documents_acc = acc_compute(model, documents_set, target_class="benign", name="documents")
    hard_negatives_acc = acc_compute(model, hard_negatives_set, target_class="benign", name="hard_negatives")
    public_prompt_injection_acc = acc_compute(model, public_prompt_injection_set, target_class="injection", name="public_prompt_injection")
    internal_prompt_injection_acc = acc_compute(model, internal_prompt_injection_set, target_class="injection", name="internal_prompt_injection")
    jailbreak_acc = acc_compute(model, jailbreak_set, target_class="injection", name="jailbreak")

    overall_acc = mean_defined([
        chat_acc,
        documents_acc,
        hard_negatives_acc,
        public_prompt_injection_acc,
        internal_prompt_injection_acc,
        jailbreak_acc,
    ])
    benign_acc = mean_defined([chat_acc, documents_acc, hard_negatives_acc])
    injection_acc = mean_defined([
        public_prompt_injection_acc,
        internal_prompt_injection_acc,
        jailbreak_acc,
    ])
    print(f"benign accuracy: {benign_acc}")
    print(f"injection accuracy: {injection_acc}")
    print(f"overall accuracy: {overall_acc}")
    return overall_acc, benign_acc, injection_acc

def wildguard_eval(model, dataset_root="datasets"):
    benign_set = []
    with open(os.path.join(dataset_root, "wildguard.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
        benign_set.append(sample["prompt"])

    wildguard_acc = acc_compute(model, benign_set, target_class="benign", name="wildguard")
    return wildguard_acc

def BIPIA_eval(model, dataset_root="datasets"):
    injection_set = []
    with open(os.path.join(dataset_root, "BIPIA_text.json"), "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
        for context in valid_dataset[key]:
            injection_set.append(context)

    BIPIA_text_acc = acc_compute(model, injection_set, target_class="injection", name="BIPIA_text")

    injection_set = []
    with open(os.path.join(dataset_root, "BIPIA_code.json"), "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
        for context in valid_dataset[key]:
            injection_set.append(context)

    BIPIA_code_acc = acc_compute(model, injection_set, target_class="injection", name="BIPIA_code")

    BIPIA_overall_acc = mean_defined([BIPIA_text_acc, BIPIA_code_acc])
    print(f"BIPIA overall accuracy: {BIPIA_overall_acc}")

    return BIPIA_overall_acc, BIPIA_text_acc, BIPIA_code_acc

def NotInject_eval(model, dataset_root="datasets/NotInject_one"):
    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_one.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
            benign_set.append(sample["prompt"])

    one_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_one")

    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_two.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
            benign_set.append(sample["prompt"])

    two_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_two")

    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_three.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
            benign_set.append(sample["prompt"])

    three_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_three")

    overall_acc = mean_defined([one_acc, two_acc, three_acc])
    print(f"NotInject overall accuracy: {overall_acc}")

    return overall_acc, one_acc, two_acc, three_acc

def evaluate(model, dataset_root):
    pint_path = os.path.join(dataset_root, "PINT.json")
    if os.path.exists(pint_path):
        pint_acc, pint_benign_acc, pint_injection_acc = pint_evaluate(model, dataset_root)
    else:
        print(f"WARNING: PINT.json not found at {pint_path}; skipping PINT evaluation.")
        pint_acc = None
        pint_benign_acc = None
        pint_injection_acc = None

    wild_acc = wildguard_eval(model, dataset_root)
    BIPIA_acc, BIPIA_text_acc, BIPIA_code_acc = BIPIA_eval(model, dataset_root)
    Notinject_acc, Notinject_one_acc, Notinject_two_acc, Notinject_three_acc = NotInject_eval(model, dataset_root)

    benign_acc = mean_defined([pint_benign_acc, wild_acc])
    injection_acc = mean_defined([pint_injection_acc, BIPIA_acc])
    overall_acc = mean_defined([Notinject_acc, benign_acc, injection_acc])

    print(f"================================ The Results ================================")
    print(f"Over-defense ACC: {Notinject_acc}")
    print(f"Benign ACC: {benign_acc}")
    print(f"Injection ACC: {injection_acc}")
    print(f"Overall ACC: {overall_acc}")


if __name__ == "__main__":
    global logger
    args = parse_args()

    set_seed(args)
    logger = get_logger(os.path.join(args.logs, "log_{}.txt".format(args.name)))

    logger.info("Effective parameters:")

    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PIGuard('microsoft/deberta-v3-base', num_labels=2, device=device)
    model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    model.to(device)

    dataset_root = args.dataset_root
    evaluate(model, dataset_root)
