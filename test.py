"""Cream
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.multiprocessing
from nltk import edit_distance
from sconf import Config
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, ".")
from cream import CreamConfig, CreamDataset, CreamModel

torch.multiprocessing.set_sharing_strategy("file_system")


def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)


def get_ED(gt, pred, case_sensitive=False):
    if case_sensitive:
        return edit_distance(pred, gt) / max(len(pred), len(gt))
    return edit_distance(pred.lower(), gt.lower()) / max(len(pred), len(gt))


def get_anls(gt, pred):
    threshold = 0.5
    ed = get_ED(gt, pred)
    score = 1 - ed if ed < threshold else 0
    return score


def test(args):
    if args.pretrained_checkpoint_path:
        src_path = Path("/".join(args.pretrained_checkpoint_path.split("/")[:-1]))
        config = json.load(open(src_path / "config.json"))
        pretrained_model = CreamModel(
            config=CreamConfig(
                decoder_layers=config["decoder_layers"],
                encoder_layers=config["encoder_layers"],
                vision_layers=config["vision_layers"],
                max_patches=config["max_patches"],
                max_enc_position_embeddings=config["max_enc_position_embeddings"],
                max_position_embeddings=config["max_position_embeddings"],
            ),
            skip_custom_init=True,
        )
        pretrained_model.load_state_dict(
            {
                k[6:].replace("_orig_mod.", ""): v
                for k, v in torch.load(args.pretrained_checkpoint_path)["state_dict"].items()
            },
            strict=True,
        )
    else:
        assert args.pretrained_checkpoint_path is None
        assert args.pretrained_model_name_or_path is not None
        pretrained_model = CreamModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")
        pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    edit_distances = []
    anls_scores = []
    predictions = []
    question_ids = []

    dataset = CreamDataset(
        dataset_name_or_paths=[args.dataset_name_or_path],
        dataset_type=args.dataset_type,
        model=pretrained_model,
        split=args.split,
        fix_seed_for_test=True,
        qa_config={"max_cl_num_words": 0},
    )

    if args.num_process > 1:
        chunk_idxes = list(range(0, len(dataset), int(len(dataset) / args.num_process) + 1)) + [len(dataset)]
        s, e = list(zip(chunk_idxes[:-1], chunk_idxes[1:]))[args.process_idx]
        print(f"last index for process_id {args.process_idx}: ", e - 1)
        dataset_sample = Subset(dataset, range(s, e))
    else:
        dataset_sample = dataset

    loader = iter(
        DataLoader(
            dataset_sample,
            batch_size=args.batch_size,
            shuffle=False,
        )
    )

    for batch in tqdm(loader):
        (
            enc_input_ids,
            enc_input_x_pos_ids,
            enc_input_y_pos_ids,
            enc_input_type_ids,
            enc_attention_mask,
            image_tensors,
            img_attention_mask,
            dec_input_ids,
            queries,
            decoder_label_ids,
            answers,
            dataset_idx,
            level,
            is_masked_text_visible,
            prompt_end_idxes,
            _question_ids,
        ) = (i.cuda() if type(i) == torch.Tensor and torch.cuda.is_available() else i for i in batch)

        if torch.cuda.is_available():
            image_tensors = image_tensors.half()

        for i in range(len(enc_input_ids)):
            assert dec_input_ids[i][prompt_end_idxes[i]] == pretrained_model.tokenizer.eos_token_id
            prediction = pretrained_model.inference(
                input_ids=enc_input_ids[i].unsqueeze(0),
                input_x_pos_ids=enc_input_x_pos_ids[i].unsqueeze(0),
                input_y_pos_ids=enc_input_y_pos_ids[i].unsqueeze(0),
                input_type_ids=enc_input_type_ids[i].unsqueeze(0),
                attention_mask=enc_attention_mask[i].unsqueeze(0),
                image_input_tensors=image_tensors[i].unsqueeze(0),
                image_attention_mask=img_attention_mask[i].unsqueeze(0),
                decoder_input_ids=dec_input_ids[i][: prompt_end_idxes[i] + 1].unsqueeze(0),
                infer_mode=args.infer_mode,
            )
            pred = prediction["predictions"][0]
            gt = eval(answers[i])[0]
            query = queries[i]

            predictions.append(pred)
            question_ids.append(_question_ids[i])
            print(f"Question ID       : {question_ids[i]}")
            print(f"Question          : {query}")
            print(f"Prediction        : {pred}")
            print(f"GT                : {gt}\n")

            edit_distances.append(get_ED(gt, pred))
            anls_scores.append(get_anls(gt, pred))

    scores = {
        "normalized_edit_distances": edit_distances,
        "mean_of_normalized_edit_distances": np.mean(edit_distances),
        "mean_of_anls": np.mean(anls_scores),
    }

    if args.save_path:
        scores["predictions"] = predictions
        save_json(
            args.save_path + f"{args.process_idx}.submit.json",
            [{"questionId": int(q_id), "answer": pred} for q_id, pred in zip(question_ids, predictions)],
        )

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint_path", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="default")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--process_idx", type=int, default=0)
    parser.add_argument("--merge_submission", action="store_true", default=False)
    parser.add_argument("--infer_mode", type=str, default="vl")
    args, left_argv = parser.parse_known_args()

    assert args.split in {"validation", "test"}

    if args.merge_submission:
        all_results = []
        for i in range(args.num_process):
            if os.path.exists(args.save_path + f"{i}.submit.json"):
                saved_result = json.load(open(args.save_path + f"{i}.submit.json"))
                all_results.extend(saved_result)
        save_json(
            args.save_path + f".submit.json",
            all_results,
        )
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        test(args)
