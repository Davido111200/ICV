import argparse
import json
import random
import sys
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm

sys.path.append("/home/s223540177/ICV")

from tasks import load_task
from models import build_model, build_tokenizer
from reinforcement.memenv import MemEnv
from mpl_toolkits.mplot3d import Axes3D


def get_args():
    parser = argparse.ArgumentParser(description='visualize')

    parser.add_argument("--model_name", choices=["llama-2"], type=str, default="llama-2")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="paradetox")
    parser.add_argument("--n_memory_samples", type=int, default=100)
    parser.add_argument("--n_neighbors", type=int, default=5)

    parser.add_argument("--prompt_version", type=str, default="default")

    parser.add_argument("--model_size", default="7b", type=str)
    parser.add_argument("--in_8bit", default=True, action="store_true")

    parser.add_argument("--distance_metric", type=str, default="euclidean")

    parser.add_argument("--n_samples", type=int, default=100)

    return parser.parse_args()


def extract_og_text(text):
    start = text.find('"') + 1  # Find the first quote
    end = text.find('"', start)  # Find the closing quote
    extracted = text[start:end]

    return extracted



def main():
    args = get_args()
    name_to_icv_dir = {
        "llama-2": "llama-2_7b"
    }
    memory_path = f"/home/s223540177/ICV/results/{args.model_name}/{args.dataset}/{args.dataset}_results_samples{args.n_memory_samples}_neighbors{args.n_neighbors}_normalized_{args.distance_metric}.json"
    icv_path = f"/home/s223540177/ICV/logger/main/{args.dataset}/seed0_default_{name_to_icv_dir[args.model_name]}_random{args.n_memory_samples}_icvstrength0.1.json"

    with open(memory_path, "rb") as file:
        # this file contains lines of dictionaries
        memory_results = [json.loads(line) for line in file]
    
    with open(icv_path, "rb") as file:
        icv_results = [json.loads(line) for line in file]

    TaskHandler = load_task(args.dataset)
    task_agent = TaskHandler(args.prompt_version)
    task_agent.set_seed(args.seed)
    task_agent.do_load()

    padding_side = "right"
    tokenizer = build_tokenizer(args.model_name, args.model_size, padding_side=padding_side)

    dataset, _ = task_agent.mk_result_dataset(tokenizer, no_padding=True, prefix='Please paraphrase the following sentence.\n ')


    if args.model_name not in ["llama-2", "falcon"]:
        raise ValueError(f"Unknown model name: {args.model_name}")
    else:
        print("Using model: ", args.model_name)

    padding_side = 'right'

    tokenizer = build_tokenizer(args.model_name, args.model_size, padding_side=padding_side)
    model = build_model(args.model_name, args.model_size, args.in_8bit)

    TaskHandler = load_task(args.dataset)
    task_agent = TaskHandler(args.prompt_version)
    task_agent.set_seed(args.seed)
    task_agent.do_load()

    eval_dataset, _ = task_agent.mk_result_dataset(tokenizer, no_padding=True, prefix='Please paraphrase the following sentence.\n ')
    train_dataset, raw_train_data = task_agent.mk_sample_dataset(tokenizer, args.n_memory_samples, seed=args.seed, no_padding=True, prefix='Please paraphrase the following sentence.\n ')

    # setting seed for random selection
    random.seed(args.seed)
    env = MemEnv(args, args.model_name, model, tokenizer, train_dataset, eval_dataset, raw_train_data)

    og_embds = []
    memory_embds = []
    icv_embds = []
    golden_embds = []

    print("Memory results: ", len(memory_results))
    print("ICV results: ", len(icv_results))

    assert len(memory_results) == len(icv_results)

    pbar = tqdm(total=len(memory_results))
    for i in range(len(memory_results)):
        memory_embds.append(env._get_hidden_state(memory_results[i]['generation']))
        icv_embds.append(env._get_hidden_state(icv_results[i]['generation']))
        golden_embds.append(env._get_hidden_state(icv_results[i]['gold']))
        og_embds.append(env._get_hidden_state(extract_og_text(tokenizer.decode(dataset[i][0]))))
        if i == args.n_samples:
            break
        pbar.update(1)

    embs1_stacked = torch.stack(memory_embds)  # Shape: (n1, d)
    embs2_stacked = torch.stack(icv_embds)  # Shape: (n2, d)
    golden_stacked = torch.stack(golden_embds)  # Shape: (n3, d)
    og_stacked = torch.stack(og_embds)

    # Concatenate all the stacked tensors for PCA
    # data = torch.cat([embs1_stacked, embs2_stacked, golden_stacked], dim=0)  # Shape: (n1 + n2 + n3, d)
    data = torch.cat([embs1_stacked, embs2_stacked, golden_stacked, og_stacked], dim=0)  # Shape: (n1 + n2 + n3, d)

    # Check the shape of the data
    print(data.shape)

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)

    # Split the transformed data back into separate groups
    n1 = embs1_stacked.shape[0]
    n2 = embs2_stacked.shape[0]
    n3 = golden_stacked.shape[0]
    n4 = og_stacked.shape[0]

    # transformed_embs1 = transformed_data[:n1]
    # transformed_embs2 = transformed_data[n1:n1 + n2]
    # transformed_golden = transformed_data[n1 + n2:]


    transformed_embs1 = transformed_data[:n1]
    transformed_embs2 = transformed_data[n1:n1 + n2]
    transformed_golden = transformed_data[n1 + n2:n1 + n2 + n3]
    transformed_og = transformed_data[n1 + n2 + n3:]

    # Plot the data with different colors
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_embs1[:, 0], transformed_embs1[:, 1], color='green', alpha=0.2, label='Memory', s=5)
    plt.scatter(transformed_embs2[:, 0], transformed_embs2[:, 1], color='red', alpha=0.5, label='ICV', s=5)
    plt.scatter(transformed_golden[:, 0], transformed_golden[:, 1], color='gold', alpha=0.3, label='Golden', s=5)
    plt.scatter(transformed_og[:, 0], transformed_og[:, 1], color='blue', alpha=0.2, label='OG', s=5)

    # Add axes and grid
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f'PCA Scatter Plot: memory_samples: {args.n_memory_samples} with {args.n_samples} samples')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    # # Create a 3D scatter plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the data with different colors
    # ax.scatter(transformed_embs1[:, 0], transformed_embs1[:, 1], transformed_embs1[:, 2], 
    #         color='green', label='Memory', alpha=0.1, s=5)
    # ax.scatter(transformed_embs2[:, 0], transformed_embs2[:, 1], transformed_embs2[:, 2], 
    #         color='red', label='ICV', alpha=0.7, s=5)
    # ax.scatter(transformed_golden[:, 0], transformed_golden[:, 1], transformed_golden[:, 2], 
    #         color='gold', label='Golden', alpha=0.2, s=5)
    # ax.scatter(transformed_og[:, 0], transformed_og[:, 1], transformed_og[:, 2], 
    #         color='blue', label='OG', alpha=0.1, s=5)

    # # Add axes and grid
    # ax.set_title(f'3D PCA Scatter Plot of {args.n_samples} samples for {args.dataset}')
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # ax.legend()
    # plt.show()


    # Save the plot to a file
    plt.savefig(f"/home/s223540177/ICV/imgs/pca/{args.dataset}/pca_plot_{args.n_memory_samples}{args.distance_metric}.png")


if __name__ == "__main__":
    main()