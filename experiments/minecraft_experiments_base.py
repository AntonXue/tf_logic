import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import HfArgumentParser, AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding                                # For one-shot string
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling                                      # For autoreg_ksteps
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments    # For seq2seq
import wandb
import json
import evaluate
from torch.utils.data import DataLoader

""" Our imports """
from common import *    # Critical definitions and path inserts
from models import AutoTaskModel
from my_datasets import *
from minecraft.dataset import MinecraftAutoregKStepsNVarsDataset
from utils.metrics import *
from utils.model_loader_utils import load_checkpoint_from_wandb
from datasets import Dataset

torch.backends.cuda.matmul.allow_tf32 = True

""" Parser for Hugging Face """

@dataclass
class MinecraftExperimentsArguments:
    """ This class captures ALL the possible minecraft experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "minecraft_experiments")),
        metadata = {"help": "Output directory of minecraft experiments."}
    )

    syn_exp_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The experiment to run."}
    )

    """ Model details """

    model_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The seq2seq model to use."}
    )

    use_pretrained : Optional[bool] = field(
        default = True,
        metadata = {"help": "Weights from the pretrained model are loaded if True. " + \
                    "Note that this restricts changes to the model " + \
                    "(default num_heads, num_layers, etc. will be used)."}
    )

    max_sequence_length: Optional[int] = field(
        default = 1024,
        metadata = {"help": "The maximum sequence length to use."} 
    )

    stop_generation_on_eos: Optional[bool] = field(
        default = False,
        metadata = {"help": "Stop generation when the EOS sequence is generated. This sequence is 'I cannot create any other items' for text and '[STATES_END]' for tags."}
    )

    """ Dataset details """

    example_format: Optional[str] = field(
        default = "text",
        metadata = {"help": "The representation of the dataset. Can be one of {'text', 'tags'}."}
    )

    num_steps: Optional[int] = field(
        default = 3,
        metadata = {"help": "The number of steps; used for autoreg_ksteps."}
    )

    min_num_vars: Optional[int] = field(
        default = 16,
        metadata = {"help": "The minimum number of propositional variables to use."}
    )

    max_num_vars: Optional[int] = field(
        default = 32,
        metadata = {"help": "The maximum number of propositional variables to use."}
    )

    train_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the training dataset."}
    )

    eval_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the eval (i.e., validation) dataset."}
    )

    max_num_distractors_triggerable: Optional[int] = field(
        default = 3,
        metadata = {"help": "The maximum number of distractors that can be triggerable."}
    )

    depth_parallel: Optional[bool] = field(
        default = True,
        metadata = {"help": "If the text should reflect parallel processing of facts at a given depth."}
    )

    """ Training details """

    train_batch_size: Optional[int] = field(
        default = 8,
        metadata = {"help": "The train batch size."}
    )

    eval_batch_size: Optional[int] = field(
        default = 8,
        metadata = {"help": "The eval (i.e., validation) batch size."}
    )

    auto_find_batch_size: Optional[bool] = field(
        default = True,
        metadata = {"help": "Automatically scale batch size if it encounters out-of-memory."}
    )

    num_epochs: Optional[int] = field(
        default = 100,
        metadata = {"help": "The number of epochs for training."}
    )

    seed: Optional[int] = field(
        default = 1234,
        metadata = {"help": "RNG seed"}
    )

    logging_steps: int = field(
        default = 10,
        metadata = {"help": "How often the HF's Trainer logs."}
    )


def minecraftexp_args_to_wandb_run_name(args: MinecraftExperimentsArguments):
    model_str = f"{args.model_name}" + \
                (f"_pt" if args.use_pretrained else "")

    if args.syn_exp_name == "one_shot_str":
        # MinecraftBC: Binary Classification
        return f"MinecraftBC__{model_str}__" + \
            f"_ef{args.example_format}" + \
            f"_msl{args.max_sequence_length}" + \
            f"_ns{args.num_steps}" + \
            f"_nv{args.min_num_vars}-{args.max_num_vars}" + \
            f"_mndt{args.max_num_distractors_triggerable}" + \
            f"_tbs{args.train_batch_size}_ebs{args.eval_batch_size}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"

    elif args.syn_exp_name == "autoreg_ksteps":
        # MinecraftNTP: Next Token Prediction
        # return f"MinecraftNTP_{model_str}__" + \
        return f"MinecraftNTP_LMEval_{model_str}__" + \
            f"_ef{args.example_format}" + \
            f"_msl{args.max_sequence_length}" + \
            f"_ns{args.num_steps}" + \
            f"_nv{args.min_num_vars}-{args.max_num_vars}" + \
            f"_mndt{args.max_num_distractors_triggerable}" + \
            f"_dp{args.depth_parallel}" + \
            f"_tbs{args.train_batch_size}_ebs{args.eval_batch_size}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"
    
    elif args.syn_exp_name == "seq2seq":
        # MinecraftSeq2Seq: Seq2Seq
        return f"MinecraftSeq2Seq_{model_str}__" + \
            f"_ef{args.example_format}" + \
            f"_msl{args.max_sequence_length}" + \
            f"_ns{args.num_steps}" + \
            f"_nv{args.min_num_vars}-{args.max_num_vars}" + \
            f"mndt{args.max_num_distractors_triggerable}" + \
            f"_tbs{args.train_batch_size}_ebs{args.eval_batch_size}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"

    else:
        raise ValueError(f"Unrecognized exp_name {args.syn_exp_name}")


def trainer_stats_for_wandb(
    args: MinecraftExperimentsArguments,
    trainer: Trainer
):
    """ Make some statistics to report to wandb """
    if args.syn_exp_name == "one_shot_str":
        num_train_qeds = torch.tensor(0)
        for i in range(len(trainer.train_dataset)):
            num_train_qeds += trainer.train_dataset[i]["label"]

        num_eval_qeds = torch.tensor(0)
        for i in range(len(trainer.eval_dataset)):
            num_eval_qeds += trainer.eval_dataset[i]["label"]

        return {
            "train_len": len(trainer.train_dataset),
            "train_qeds": num_train_qeds.item(),
            "eval_len" : len(trainer.eval_dataset),
            "eval_qeds": num_eval_qeds.item()
        }

    elif args.syn_exp_name == "autoreg_ksteps":
        return {
            "train_len": len(trainer.train_dataset),
            "eval_len": len(trainer.eval_dataset),
        }
    
    elif args.syn_exp_name == "seq2seq":
        return {
            "train_len": len(trainer.train_dataset),
            "eval_len": len(trainer.eval_dataset),
        }

    else:
        raise ValueError(f"Unrecognized exp_name {args.syn_exp_name}")

# TODO: Refactor this later (move to common.py)
def set_wandb_run_id(args: MinecraftExperimentsArguments):
    """ Set the wandb run id for this experiment """
    run_id_map_file = Path(DUMP_DIR, "run_id_map.json")
    if not run_id_map_file.exists():
        run_id_map = {}
    else:
        run_id_map = json.load(open(str(run_id_map_file), "r"))

    exp_id = minecraftexp_args_to_wandb_run_name(args)
    
    if exp_id in run_id_map.keys():
        run_id = run_id_map[exp_id]
        print(f"Found run_id {run_id} for {exp_id}")
    else:
        run_id = wandb.util.generate_id()
        run_id_map[exp_id] = run_id
        json.dump(run_id_map, open(str(Path(DUMP_DIR, "run_id_map.json")), "w"))
        print(f"Generated run_id {run_id} for {exp_id}")
    
    # Set the run_id to ensure resuming any previous run
    os.environ["WANDB_RUN_ID"] = run_id

class TrainerForNextTokenPrediction(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        self._memory_tracker.start()

        self.model.eval()
        n_contains_target = 0
        n_contains_create_target = 0
        n_contains_create_all_targets_in_labels = 0
        n_complete_prompt = 0
        n_complete_generation = 0

        eval_loader = DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        with torch.no_grad():
            for batch in eval_loader:
                preds = self.model.generate(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(self.args.device),
                    # max_new_tokens=512,
                    max_length=1024,
                )
                # See if the preds match the labels
                preds = preds.cpu().numpy()
                preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

                decoded_labels = batch["labels"]
                decoded_input_ids = batch["data"]
                    

                # Log some samples
                for i in range(min(2, len(batch["input_ids"]))):
                    print(f"prompt: {decoded_input_ids[i]}")
                    print(f"preds: {decoded_preds[i]}")
                    print(f"generation: {decoded_preds[i][len(decoded_input_ids[i]):]}")
                    print(f"labels: {decoded_labels[i]}")
                    print(f"target: {batch['target'][i]}")
                    print(f"qed: {batch['qed'][i]}")
                    print("----------------------------")

                # Count the number of examples where the preds contain the target if batch["qed"] == 1
                # Do this for the entire batch
                for i in range(len(batch["input_ids"])):
                    target = batch["target"][i]
                    generation = decoded_preds[i][len(decoded_input_ids[i]):]
                    if batch["qed"][i] == 1:
                        if target in generation:
                            n_contains_target += 1
                        if f"so I can create {target}" in generation or f"-> {target}" in generation:
                            n_contains_create_target += 1
                    else:
                        if target not in generation:
                            n_contains_target += 1
                        if f"so I can create {target}" not in generation and f"->  {target}" not in generation:
                            n_contains_create_target += 1

                    # Count the number of examples where the generation is complete
                    if "I cannot create any other items" in generation or "[STATES_END]" in generation:
                        n_complete_generation += 1
                    else:
                        # print the generation
                        print("============= Generation is incomplete =============")
                        print(f"generation: {generation}")
                        print(f"labels: {decoded_labels[i]}")
                        print(f"prompt: {decoded_input_ids[i]}")
                        print("===================================================")

                    # Count the number of examples where the prompt is complete
                    if "I can create the following items:" in decoded_input_ids[i] or "[STATES_START]" in decoded_input_ids[i]:
                        n_complete_prompt += 1

                    # Count the number of examples where all the "so I can create x" in the labels are in the generation
                    label = decoded_labels[i]
                    all_targets_in_labels = [t.split(".")[0].strip() for t in label.split("so I can create ")[1:]]
                    all_targets_in_labels = set(["so I can create " + target for target in all_targets_in_labels])
                    all_targets_in_generation = [t.split(".")[0].strip() for t in generation.split("so I can create ")[1:]]
                    all_targets_in_generation = set(["so I can create " + target for target in all_targets_in_generation])

                    # Check if the two lists are equal
                    if all_targets_in_labels == all_targets_in_generation:
                        n_contains_create_all_targets_in_labels += 1
                    else:
                        print("============= Label and Generation mismatch =============")
                        print(f"all_targets_in_labels: {all_targets_in_labels}")
                        print(f"all_targets_in_generation: {all_targets_in_generation}")
                        print(f"labels: {label}")
                        print(f"generation: {generation}")
                        print("========================================================")

            metrics = {
                "n_contains_target": n_contains_target,
                "n_contains_create_target": n_contains_create_target,
                "n_contains_create_all_targets_in_labels": n_contains_create_all_targets_in_labels,
                "n_complete_prompt": n_complete_prompt,
                "n_complete_generation": n_complete_generation,
                "len_eval_dataset": len(self.eval_dataset),
                "contains_target": n_contains_target / len(self.eval_dataset),
                "contains_create_target": n_contains_create_target / len(self.eval_dataset),
                "complete_prompt": n_complete_prompt / len(self.eval_dataset),
                "complete_generation": n_complete_generation / len(self.eval_dataset),
                "contains_create_all_targets_in_labels": n_contains_create_all_targets_in_labels / len(self.eval_dataset),
                }
            
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            
            print(metrics)
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)
            return metrics

def make_trainer_for_synthetic(
    args: MinecraftExperimentsArguments,
    report_to: str = "wandb"
):
    """ Make a Hugging Face Trainer object """

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.example_format == "tags":
        new_tokens = ["[RULES_START]", "[RULES_END]", "[FACTS_START]", "[FACTS_END]", "[STATES_START]", "[STATES_END]"]
        new_tokens += ["->"]
        new_tokens += ["+"]
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(item):
        return tokenizer(item["data"], truncation=True, padding="max_length", max_length=args.max_sequence_length)
        return tokenizer(item["data"], truncation=True, padding=True)
    
    # Add labels to the datasets where the labels are the same as the input ids
    # This is because the model is autoregressive and the labels are the same as the input
    def add_labels_to_examples(examples):
        if "labels" not in examples:
            examples["labels"] = examples["input_ids"].clone()
        # else:
        #     labels = tokenizer(examples["labels"], truncation=True, padding="max_length", max_length=args.max_sequence_length)
        #     # labels = tokenizer(examples["labels"], truncation=True, padding=True)
        #     examples["labels"] = labels["input_ids"]
        return examples
    
    metric = evaluate.load("exact_match")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        print("preds.shape: ", preds.shape)
        print("labels.shape: ", labels.shape)

        print("preds[0]: ", preds[0])
        print("labels[0]: ", labels[0])

        # Batch decode both the predictions and the labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log the first few examples
        for i in range(5):
            print(f"preds: {decoded_preds[i]}")
            print(f"labels: {decoded_labels[i]}")
            print("----------------------------")

        score = metric.compute(predictions=decoded_preds, references=decoded_labels)
        print(score)
        return score

    if args.syn_exp_name == "one_shot_str":
        print("Creating a trainer for Binary Classification")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        big_dataset = MinecraftAutoregKStepsNVarsDataset(
            num_steps=args.num_steps,
            num_vars_range=(args.min_num_vars, args.max_num_vars),
            dataset_len=args.train_len + args.eval_len,
            tokenizer=tokenizer,
            task_type="binary_classification",
            example_format=args.example_format,
            max_num_distractors_triggerable=args.max_num_distractors_triggerable)
        
        train_len, eval_len = args.train_len, args.eval_len
        if len(big_dataset) < args.train_len + args.eval_len:
            train_len, eval_len = len(big_dataset) * (args.train_len / (args.train_len + args.eval_len)), len(big_dataset) * (args.eval_len / (args.train_len + args.eval_len))
            train_len, eval_len = int(train_len), len(big_dataset) - int(train_len)

        print(f"train_len: {train_len}, eval_len: {eval_len}")
        
        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [train_len, eval_len])
        
        train_hf_dataset = Dataset.from_dict({
            "data": [train_dataset[i]['data'] for i in range(len(train_dataset))],
            "label": [train_dataset[i]['labels'] for i in range(len(train_dataset))],
        }).with_format("torch")

        eval_hf_dataset = Dataset.from_dict({
            "data": [eval_dataset[i]['data'] for i in range(len(eval_dataset))],
            "label": [eval_dataset[i]['labels'] for i in range(len(eval_dataset))],
        }).with_format("torch")

        print("Created HF datasets")
    
        train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
        # eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_hf_dataset

        print("Tokenized HF datasets")

        tfl_model = AutoModelForSequenceClassification.from_pretrained(
                        args.model_name, num_labels=2
                    )
        tfl_model.config.pad_token_id = tokenizer.pad_token_id

        training_args = TrainingArguments(
            str(Path(args.output_dir, minecraftexp_args_to_wandb_run_name(args))),
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = minecraftexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.10,
            save_strategy = "no",
            save_safetensors = False)

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = one_shot_metrics)
    
    elif args.syn_exp_name == "autoreg_ksteps":
        print("Creating a trainer for Next Token Prediction")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        big_dataset = MinecraftAutoregKStepsNVarsDataset(
            num_steps=args.num_steps,
            num_vars_range=(args.min_num_vars, args.max_num_vars),
            dataset_len=args.train_len + args.eval_len,
            tokenizer=tokenizer,
            task_type="next_token_prediction",
            example_format=args.example_format,
            max_num_distractors_triggerable=args.max_num_distractors_triggerable,
            depth_parallel=args.depth_parallel)
        
        train_len, eval_len = args.train_len, args.eval_len
        if len(big_dataset) < args.train_len + args.eval_len:
            train_len, eval_len = len(big_dataset) * (args.train_len / (args.train_len + args.eval_len)), len(big_dataset) * (args.eval_len / (args.train_len + args.eval_len))
            train_len, eval_len = int(train_len), len(big_dataset) - int(train_len)

        print(f"train_len: {train_len}, eval_len: {eval_len}")
        
        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [train_len, eval_len])
        
        train_hf_dataset = Dataset.from_dict({
            "data": [train_dataset[i]['data'] + tokenizer.eos_token for i in range(len(train_dataset))],
            # "data": [train_dataset[i]['data'] for i in range(len(train_dataset))],
        }).with_format("torch")

        eval_hf_dataset = Dataset.from_dict({
            "data": [eval_dataset[i]['data'] + tokenizer.eos_token for i in range(len(eval_dataset))],
            # "data": [eval_dataset[i]['data'] for i in range(len(eval_dataset))],
        }).with_format("torch")

        # # TODO: Move the labels logic to the dataset
        # if args.example_format == "tags":
        #     eval_hf_dataset = Dataset.from_dict({
        #         "data": [eval_dataset[i]['data'].split("[STATES_START]")[0].strip() + "[STATES_START]" for i in range(len(eval_dataset))],
        #         "labels": [eval_dataset[i]['data'].split("[STATES_START]")[1] for i in range(len(eval_dataset))],
        #         "target": [eval_dataset[i]["target"] for i in range(len(eval_dataset))],
        #         "qed": [eval_dataset[i]["labels"] for i in range(len(eval_dataset))],
        #     }).with_format("torch")
        # else:
        #     eval_hf_dataset = Dataset.from_dict({
        #         "data": ["\n".join(eval_dataset[i]['data'].split("\n")[:5]) for i in range(len(eval_dataset))],
        #         "labels": ["\n".join(eval_dataset[i]['data'].split("\n")[5:]) for i in range(len(eval_dataset))],
        #         "target": [eval_dataset[i]["target"] for i in range(len(eval_dataset))],
        #         "qed": [eval_dataset[i]["labels"] for i in range(len(eval_dataset))],
        #     }).with_format("torch")

        # # Print qed distribution for the eval dataset
        # print("Eval dataset qed distribution: ", eval_hf_dataset["qed"].numpy().tolist().count(1) / len(eval_hf_dataset))

        print("Created HF datasets")

        if args.example_format == "tags":
            # Add all space-separated tokens to the tokenizer
            more_new_tokens = set(" ".join([train_dataset[i]['data'] for i in range(len(train_dataset))]).split())
            more_new_tokens = more_new_tokens.union(set(" ".join([eval_dataset[i]['data'] for i in range(len(eval_dataset))]).split()))
            # Remove , from the tokens
            more_new_tokens = set([token.replace(",", "") for token in more_new_tokens])
            more_new_tokens.add(",")
            more_new_tokens = more_new_tokens - set(tokenizer.vocab.keys())
            tokenizer.add_tokens(list(more_new_tokens))

        train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

        print("Tokenized HF datasets")
        
        train_dataset = train_dataset.map(add_labels_to_examples, batched=True)
        eval_dataset = eval_dataset.map(add_labels_to_examples, batched=True)

        tfl_model = AutoModelForCausalLM.from_pretrained(args.model_name, pad_token_id=tokenizer.eos_token_id)
        tfl_model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            str(Path(args.output_dir, minecraftexp_args_to_wandb_run_name(args))),
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            gradient_accumulation_steps = 4,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = minecraftexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.10,
            save_strategy = "epoch",
            save_total_limit = 2,
            save_safetensors = False)
        
        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
        )
        
        # return TrainerForNextTokenPrediction(
        #     tfl_model,
        #     training_args,
        #     train_dataset = train_dataset,
        #     eval_dataset = eval_dataset,
        #     tokenizer = tokenizer,
        #     data_collator = data_collator)

    
    elif args.syn_exp_name == "seq2seq":
        print("Creating a trainer for Seq2Seq")
    
        big_dataset = MinecraftAutoregKStepsNVarsDataset(
            num_steps=args.num_steps,
            num_vars_range=(args.min_num_vars, args.max_num_vars),
            dataset_len=args.train_len + args.eval_len,
            tokenizer=tokenizer,
            task_type="next_token_prediction",
            example_format=args.example_format,
            max_num_distractors_triggerable=args.max_num_distractors_triggerable)
        
        train_len, eval_len = args.train_len, args.eval_len
        if len(big_dataset) < args.train_len + args.eval_len:
            train_len, eval_len = len(big_dataset) * (args.train_len / (args.train_len + args.eval_len)), len(big_dataset) * (args.eval_len / (args.train_len + args.eval_len))
            train_len, eval_len = int(train_len), len(big_dataset) - int(train_len)

        print(f"train_len: {train_len}, eval_len: {eval_len}")
        
        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [train_len, eval_len])
        
        train_hf_dataset = Dataset.from_dict({
            "data": [train_dataset[i]['data'].split("[STATES_START]")[0].strip() for i in range(len(train_dataset))],
            "labels": ["[STATES_START]" + train_dataset[i]['data'].split("[STATES_START]")[1] for i in range(len(train_dataset))],
        }).with_format("torch")

        eval_hf_dataset = Dataset.from_dict({
            "data": [eval_dataset[i]['data'].split("[STATES_START]")[0].strip() for i in range(len(eval_dataset))],
            "labels": ["[STATES_START]" + eval_dataset[i]['data'].split("[STATES_START]")[1] for i in range(len(eval_dataset))],
        }).with_format("torch")

        print("Created HF datasets")
    
        train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

        print("Tokenized HF datasets")
        
        train_dataset = train_dataset.map(add_labels_to_examples, batched=True)
        eval_dataset = eval_dataset.map(add_labels_to_examples, batched=True)

        tfl_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, pad_token_id=tokenizer.eos_token_id)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=tfl_model)

        training_args = Seq2SeqTrainingArguments(
            str(Path(args.output_dir, minecraftexp_args_to_wandb_run_name(args))),
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = minecraftexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.10,
            save_strategy = "no",
            save_safetensors = False,
            predict_with_generate=True)
        
        return Seq2SeqTrainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = compute_metrics)

    else:
        raise ValueError(f"Unrecognized exp_name {args.syn_exp_name}")


""" Main stuff """

if __name__ == "__main__":
    parser = HfArgumentParser(MinecraftExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    trainer = make_trainer_for_synthetic(args)

    set_wandb_run_id(args)
    
    # Log some preliminary training stats
    trainer_stats = trainer_stats_for_wandb(args, trainer)

    experiment_id = minecraftexp_args_to_wandb_run_name(args)
    checkpoint = load_checkpoint_from_wandb(
        experiment_out_dir=str(Path(args.output_dir, experiment_id)),
        experiment_id=experiment_id
    )
    baseline_eval_results = None
    if checkpoint is not None:
        print("Found checkpoint: ", checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("No checkpoint found. Training from scratch.")
        # Evaluate the model before training and log the results
        baseline_eval_results = trainer.evaluate()
        wandb.run.summary["eval_before_train"] = baseline_eval_results
        trainer.train()

    # Compute the evaluation metrics for the last checkpoint
    eval_results = trainer.evaluate()
    print(f"Eval results: {eval_results}")
    
    
    wandb.run.summary["trainer_stats"] = trainer_stats
    wandb.finish()
