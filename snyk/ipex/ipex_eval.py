#   Copyright 2021 DeepCode AG

#   Author: Robin Staab


import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.profiler as profiler
import time
from tqdm import tqdm
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed
from utils import auxiliary
from utils import data_reader
from utils import prepare_data
from transformers.models.t5.configuration_t5 import T5Config
import torchdynamo
from typing import List
set_seed(42)

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    scripted = torch.jit.trace(gm, example_inputs)
    return scripted

def check_input_alignment(dataset: prepare_data.SplittedDataSetWithWarningIds):
    for warning in dataset.test_inputs:
        inputs = dataset.test_inputs[warning]
        infos = dataset.test_info[warning]
        for i, code in enumerate(inputs):
            assert (
                code == infos[i].get_t5_representation(True).input
            ), "the alignment between test inputs and test info is not okay"


def beam_eval_loop(
    model,
    trainer: Seq2SeqTrainer,
    dataset: prepare_data.BugFixDataset,
    num_beams: int,
    num_predictions: int,
    target_max_length: int,
) -> Tuple[np.ndarray, np.ndarray, float]:

    dataloader = trainer.get_test_dataloader(dataset)

    top_1_preds: List[torch.Tensor] = []
    top_k_preds: List[torch.Tensor] = []
    durs = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            #with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
            #    batch_beam_output_ids = model.generate(
            #        inputs["input_ids"],
            #        attention_mask=inputs["attention_mask"],
            #        max_length=target_max_length,
            #        num_beams=num_beams,
            #        early_stopping=True,
            #        num_return_sequences=num_predictions,
            #    )
            #print(prof.key_averages().table(sort_by='self_cpu_time_total'))

            t0 = time.time()
            #print(inputs["input_ids"])
            batch_beam_output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=target_max_length,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=num_predictions,
            )
            t1 = time.time()
            decoder_time = t1 - t0
            print('decoder time =', decoder_time)
            durs.append(decoder_time)

            if batch_beam_output_ids.shape[-1] < target_max_length:
                padded_output_ids = torch.zeros((batch_beam_output_ids.shape[0], target_max_length))
                padded_output_ids[:, : batch_beam_output_ids.shape[-1]] = batch_beam_output_ids
                batch_beam_output_ids = padded_output_ids

            inx = np.array([num_predictions * i for i in range(len(inputs["input_ids"]))])
            top_1_preds.append(batch_beam_output_ids[inx].detach().cpu())
            top_k_preds.append(batch_beam_output_ids.detach().cpu())

    top_1_preds_np = torch.cat(top_1_preds).numpy()
    top_k_preds_np = torch.cat(top_k_preds).numpy()

    # Remove the start token
    top_1_preds_np = np.delete(
        top_1_preds_np, 0, axis=1
    )  # delete the dummy <start_decoding> token"
    top_1_preds_np = np.insert(
        top_1_preds_np, target_max_length - 1, 0, axis=1
    )  # pad back at the end

    top_k_preds_np = np.delete(
        top_k_preds_np, 0, axis=1
    )  # delete the dummy <start_decoding> token"
    top_k_preds_np = np.insert(
        top_k_preds_np, target_max_length - 1, 0, axis=1
    )  # pad back at the end

    print('dur: {:.2f}ms'.format(sum(durs)/len(durs)*1000))
    return top_1_preds_np, top_k_preds_np, sum(durs)/len(durs)


def evaluate_data(args):
    data_test = data_reader.get_data_from_paths(args.data_files_test)
    print("Running from Hugginface checkpoint directory")
    tokenizer = T5Tokenizer.from_pretrained(args.load_model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.load_model_dir)

    if args.error_types:
        all_warning_types = args.error_types
    else:
        all_warning_types = list(set(prepare_data.extract_warning_types(data_test)))

    dataset: prepare_data.SplittedDataSetWithWarningIds = prepare_data.create_data(
        data_train=[],
        data_val=[],
        data_test=data_test,
        linter_warnings=all_warning_types,
        include_warning=True,
        model_name=args.model_name,
    )
    '''
    train_dataset = prepare_data.create_dataset(
        dataset.train_inputs,
        dataset.train_labels,
        tokenizer,
        pad_truncate=True,
        max_length=256,
    )
    val_dataset = prepare_data.create_dataset(
        dataset.val_inputs, dataset.val_labels, tokenizer, pad_truncate=True
    )
    '''
    # for new transformert, train, val dataste should not be None
    input_sequence = "HuggingFace is a company"
    output_sequence = "HuggingFace est une entreprise"

    train_dataset = prepare_data.create_dataset(
        input_sequence,
        output_sequence,
        tokenizer,
        pad_truncate=True,
        max_length=256,
    )
    
    val_dataset = prepare_data.create_dataset(
        input_sequence, output_sequence, tokenizer, pad_truncate=True
    )

    print("Creating Trainer")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        do_eval=True,
        evaluation_strategy="epoch",
        eval_accumulation_steps=args.batch_size,
        predict_with_generate=True,
    )

    # Needed for easy model wrapping and dataloader creation
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=[torch.optim.Adam(params=model.parameters()), None],
        tokenizer=tokenizer,
    )

    counter = 0
    for warning in dataset.test_inputs:
        counter += len(dataset.test_inputs[warning])
    print("Number of testing samples: ", counter)

    # test that the samples are well aligned among inputs and info
    check_input_alignment(dataset)

    sorted_warning_types = all_warning_types[:]
    sorted_warning_types = list(
        sorted(
            sorted_warning_types,
            key=lambda w: len(dataset.test_inputs[w]),
            reverse=True,
        )
    )
    for w in sorted_warning_types:
        print(w, len(dataset.test_inputs[w]))

    # Testing
    print("Testing started")
    top1_accuracies: Dict[str, float] = defaultdict(float)
    topk_accuracies: Dict[str, float] = defaultdict(float)
    num_predictions = args.num_predictions

    # model = trainer._wrap_model(model, training=False).eval()
    input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    model.eval()
    '''
    from transformers.modeling_fx_utils import symbolic_trace
    traced_model = symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            batch_size=1,
            sequence_length=128,
        )
    print(model)
    exit
    '''
    # dynamic quantization
    #model = torch.jit.script(model.eval())
    import models_utils
    '''
    for idx in range(len(model.encoder.block)):
        config = T5Config()
        attention = model.encoder.block[idx].layer[0].SelfAttention
        config.is_decoder = attention.is_decoder
        config.relative_attention_num_buckets = attention.relative_attention_num_buckets 
        config.relative_attention_max_distance = attention.relative_attention_max_distance 
        config.d_model = attention.d_model 
        config.d_kv = attention.key_value_proj_dim
        config.num_heads = attention.n_heads
        config.dropout_rate = attention.dropout
        attention_opti = models_utils.T5Attention(config, attention.has_relative_attention_bias)
                
        attention_opti.q = attention.q 
        attention_opti.k = attention.k
        attention_opti.v =  attention.v
        attention_opti.o =  attention.o
        attention_opti.qkv.weight.data = torch.cat([attention_opti.q.weight, attention_opti.k.weight, attention_opti.v.weight])
        if attention_opti.q.bias is not None:
            attention_opti.qkv.bias.data = torch.cat([attention_opti.q.bias, attention_opti.k.bias, attention_opti.v.bias])
        if attention.has_relative_attention_bias:
            attention_opti.relative_attention_bias = attention.relative_attention_bias
        attention_opti.pruned_heads = attention.pruned_heads
        attention_opti.gradient_checkpointing = attention.gradient_checkpointing

        model.encoder.block[idx].layer[0].SelfAttention = attention_opti

        # fused layer_norm
        for j in range(2):
            layer_norm =  model.encoder.block[idx].layer[j].layer_norm
            hidden_size = layer_norm.weight.size(0)
            fused_layer_norm = models_utils.T5LayerNorm(hidden_size, layer_norm.variance_epsilon)
            fused_layer_norm.weight.data = layer_norm.weight.data.clone()
            model.encoder.block[idx].layer[j].layer_norm = fused_layer_norm
    # decoder
    for idx in range(len(model.decoder.block)):
        config = T5Config()
        attention = model.decoder.block[idx].layer[0].SelfAttention
        config.is_decoder = attention.is_decoder
        config.relative_attention_num_buckets = attention.relative_attention_num_buckets 
        config.relative_attention_max_distance = attention.relative_attention_max_distance 
        config.d_model = attention.d_model 
        config.d_kv = attention.key_value_proj_dim
        config.num_heads = attention.n_heads
        config.dropout_rate = attention.dropout
        attention_opti = models_utils.T5Attention(config, attention.has_relative_attention_bias)
                
        attention_opti.q = attention.q 
        attention_opti.k = attention.k
        attention_opti.v =  attention.v
        attention_opti.o =  attention.o
        attention_opti.qkv.weight.data = torch.cat([attention_opti.q.weight, attention_opti.k.weight, attention_opti.v.weight])
        if attention_opti.q.bias is not None:
            attention_opti.qkv.bias.data = torch.cat([attention_opti.q.bias, attention_opti.k.bias, attention_opti.v.bias])
        if attention.has_relative_attention_bias:
            attention_opti.relative_attention_bias = attention.relative_attention_bias
        attention_opti.pruned_heads = attention.pruned_heads
        attention_opti.gradient_checkpointing = attention.gradient_checkpointing

        model.decoder.block[idx].layer[0].SelfAttention = attention_opti
        
        # fused layer_norm
        for j in range(3):
            layer_norm =  model.decoder.block[idx].layer[j].layer_norm
            hidden_size = layer_norm.weight.size(0)
            fused_layer_norm = models_utils.T5LayerNorm(hidden_size, layer_norm.variance_epsilon)
            fused_layer_norm.weight.data = layer_norm.weight.data.clone()
            model.decoder.block[idx].layer[j].layer_norm = fused_layer_norm

    '''
    torch.quantization.quantize_dynamic(model, inplace=True)

    if args.ipex:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model)
    model.eval()

    durs = []
    pbar = tqdm(sorted_warning_types)
    for i, warning in enumerate(pbar):
        test_warning = dataset.test_inputs[warning]
        test_warning_labels = dataset.test_labels[warning]
        test_warning_info = dataset.test_info[warning]
        target_max_length = 256  # Set to 256 if enough memory
        pbar.set_description(f"Rule {i}: {warning}, Num. samples: {len(test_warning)}")
        if len(test_warning) == 0:
            continue

        correct_counter, total_counter = 0, 0

        test_warning_batch = test_warning
        test_warning_labels_batch = test_warning_labels

        test_warning_dataset = prepare_data.create_dataset(
            test_warning_batch,
            test_warning_labels_batch,
            tokenizer,
            pad_truncate=True,
            max_length=target_max_length,
        )

        target_ids = tokenizer(
            test_warning_labels_batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=target_max_length,
        ).input_ids

        target_ids = np.array(target_ids)

        # Prediction
        top_1_preds, top_k_preds, dur = beam_eval_loop(
            model,
            trainer,
            test_warning_dataset,
            num_beams=args.beam_size,
            num_predictions=args.num_predictions,
            target_max_length=target_max_length,
        )
        exit()
        durs.append(dur)
        correct_counter += int(np.sum(np.all(np.equal(target_ids, top_1_preds), axis=1)))
        total_counter += len(top_1_preds)
        for k, output_id in enumerate(top_1_preds):
            pred = tokenizer.decode(output_id, skip_special_tokens=True)
            target = tokenizer.decode(target_ids[k], skip_special_tokens=True)
            test_warning_info[k].predictions = [pred]


        top1_acc = correct_counter / total_counter
        top1_accuracies[warning] = top1_acc

        k_target_ids = np.repeat(target_ids, num_predictions, axis=0)
        exact_matches = np.all(np.equal(k_target_ids, top_k_preds), axis=1)

        beam_exact_matches = np.reshape(exact_matches, newshape=(-1, num_predictions))
        beam_exact_matches = np.any(beam_exact_matches, axis=1)
        topk_acc = np.sum(beam_exact_matches) / float(total_counter)
        topk_accuracies[warning] = topk_acc

        print(f"Rule: {warning} Top1: {top1_acc} Topk: {topk_acc}")

        dataset.test_info[warning] = test_warning_info

    durs = durs[10:]
    print('dur: {:.2f}ms'.format(sum(durs)/len(durs)*1000))
    os.makedirs(args.out_dir, exist_ok=True)

    test_list = []
    for rule_name in dataset.test_info:
        test_list += dataset.test_info[rule_name]
    with open(os.path.join(args.out_dir, args.test_outfile), "w") as test_file:
        json.dump(
            [datapoint.__dict__ for datapoint in test_list],
            test_file,
            default=lambda o: o.__dict__,
        )

    top1_accuracies["average"] = auxiliary.get_average_of_dict(top1_accuracies)
    topk_accuracies["average"] = auxiliary.get_average_of_dict(topk_accuracies)
    auxiliary.write_scores_to_text_file(
        top1_accuracies, os.path.join(args.out_dir, "first_accs.txt")
    )
    auxiliary.write_scores_to_text_file(
        topk_accuracies, os.path.join(args.out_dir, "beam_accs.txt")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lmd",
        "--load-model-dir",
        type=str,
        default="",
        help="path to checkpoint that will be tested",
    )
    parser.add_argument(
        "-od",
        "--out-dir",
        type=str,
        default="",
        help="path to out directory",
    )
    parser.add_argument(
        "-dft",
        "--data-files-test",
        nargs="+",
        required=True,
        help="names of the data test files",
    )
    parser.add_argument(
        "-mn",
        "--model-name",
        type=str,
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        required=True,
    )
    parser.add_argument(
        "-et",
        "--error-types",
        nargs="+",
        required=False,
        help="Allows you to select specific rules used for training",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=32)
    parser.add_argument("-beams", "--beam-size", type=int, default=5)
    parser.add_argument("-nump", "--num-predictions", type=int, default=5)
    parser.add_argument("-test-o", "--test-outfile", type=str, default="test_data.json")
    parser.add_argument('--ipex', default=False, action="store_true")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_data(args)
