import gymnasium
from gymnasium import spaces
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Text,
    Tuple,
)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import evaluate
import torch
import random
from itertools import combinations
from tqdm import tqdm
from torch.nn import functional as F

from models import build_model, build_tokenizer
from common import mk_parser
from tasks import load_task

from tasks.base import obtain_memory_icv

from utils.llm_layers import add_icv_layers, remove_icv_layers
from utils.forward_tracer import ForwardTrace
from utils.forward_tracer import ForwardTracer


from LIVE.inference import generate_answers, get_icv


from parlai.utils.safety import OffensiveLanguageClassifier


class HiddenStateNormalizer:
    def __init__(self):
        self.running_mean = None
        self.running_std = None
        self.count = 0

    def update(self, embeddings):
        # Update running mean and std
        batch_mean = embeddings.mean(dim=0)
        batch_std = embeddings.std(dim=0)

        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            # Weighted average for running mean and std
            total_count = self.count + embeddings.size(0)
            self.running_mean = (self.running_mean * self.count + batch_mean * embeddings.size(0)) / total_count
            self.running_std = (self.running_std * self.count + batch_std * embeddings.size(0)) / total_count
            self.count = total_count

    def normalize(self, embeddings):
        # Normalize embeddings using running mean and std
        return (embeddings - self.running_mean) / (self.running_std + 1e-8)  # Avoid division by zero



class MemEnv(gymnasium.Env):
    def __init__(self, args, model_name, model, tokenizer, train_dataset, eval_dataset, raw_train_data):
        self.args = args
        self.model_name = model_name    
        self.model = model
        self.tokenizer = tokenizer

        self.model_shakespeare = RobertaForSequenceClassification.from_pretrained("/home/s223540177/ICV/evaluate/shakes/saves/shakespeare-classifier", num_labels=2).to("cuda")
        self.tokenizer_shakespeare = RobertaTokenizer.from_pretrained("/home/s223540177/ICV/evaluate/shakes/saves/shakespeare-classifier")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.raw_train_data = raw_train_data

        # build model and tokenizer
        model_to_hidden = {
            "llama-2": 4096,
            "falcon": 4544, 
        }
        
        model_to_n_layers = {
            "llama-2": 32,
            "falcon": 32,
        }

        self.n_layers = model_to_n_layers[self.model_name]
        self.hidden_size = model_to_hidden[self.model_name]

        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.hidden_size,),
            dtype=float
        )

        # state is discrete
        self.action_space = spaces.Discrete(1)

        self.current_data_idx = 0
        self.current_reward = 0
        self.steps = 0
        
    def _get_train_dataset(self):
        return self.train_dataset

    def _get_obs(self):
        return {"previous_actions": self.previous_actions, "current_state": self.current_state, "current_ts": self.steps}

    def _get_current_state(self):
        return self.current_state

    def _get_hidden_state(self, inputs):
        text = [inputs]  # assuming the text is in 'text' field
        encoded_inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            last_hidden_state = self.model(**encoded_inputs, output_hidden_states=True).hidden_states[-1]
        
        weights_for_non_padding = encoded_inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

        return sentence_embeddings.cpu().detach().numpy()[0]



    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        # reset the total time steps
        self.steps = 0

        # reset the previous actions
        self.previous_actions = []

        # reset previous reward
        self.current_reward = 0

        original_sentence = self.raw_train_data[self.current_data_idx]['query'][0]

        # get hidden state of the sentence
        self.current_state = self._get_hidden_state(original_sentence)

        state = self._get_current_state()
        return state, {}

    def evaluate(self, metric_name, current_state, gold_label):
        if metric_name == "bleu":
            result = self.bleu.compute(predictions=[current_state], references=[gold_label])
        elif metric_name == "rouge":
            result = self.rouge.compute(predictions=[current_state], references=[gold_label])
        elif metric_name == "bertscore":
            result = self.bertscore.compute(predictions=[current_state], references=[gold_label], lang='en')
            result = round(np.mean(result['f1']), 4)
        elif metric_name == "meteor":
            result = self.meteor.compute(predictions=[current_state], references=[gold_label])
        elif metric_name == "toxicity":
            # special case where the result is a boolean
            ppred, prob = self.toxicity.contains_offensive_language(current_state)
            if prob > 0.9 and ppred:
                result = 0.0
            else:
                result = 1.0
        elif metric_name == "shakespeare":
            inputs = self.tokenizer_shakespeare(current_state, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            outputs = self.model_shakespeare(**inputs)
            prediction = outputs.logits.argmax(axis=-1).item()
            if prediction == 0:
                # this corresponds to modern style, give reward = 0
                result = 0.0
            else:
                result = 1.0
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        return result

    # def step(self, action):
    #     original_sentence, reference = self.raw_train_data[self.current_data_idx]['query'][0], self.raw_train_data[self.current_data_idx]['query'][1]

    #     self.current_state = self._get_hidden_state(original_sentence)
        
    #     icv = obtain_memory_icv(self.model, self.train_dataset.tokenize_each_demonstration(
    #         [(original_sentence, reference)], prefix=("", "")), rank=1
    #     )
    #     icv = icv[1:]
        
    #     # add the ICV to the current state
    #     add_icv_layers(self.model, torch.stack([icv],dim=1).cuda(), [self.args.lam])

    #     if 'llama' in self.args.model_type:
    #         gen_args = {
    #             'temperature': 0.45,
    #             'do_sample': True,
    #             'top_k': 0,
    #             'top_p': 1.0,
    #             'eos_token_id': [1642, 13492, 26036, 29908,self.tokenizer.encode('.10')[-1]]
    #         }
    #     elif 'falcon' in self.args.model_type:
    #         gen_args = {
    #             'do_sample': False,
    #             'num_beams': 10,
    #             'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166]
    #         }
    #     else:
    #         gen_args = {}
        
    #     with torch.no_grad():
    #         input_id = self.train_dataset[self.current_data_idx][0].unsqueeze(0) # ensure 2d input_ids
    #         attention_mask = self.train_dataset[self.current_data_idx][1].unsqueeze(0)
    #         generation_output = self.model.generate(
    #             input_ids=input_id.cuda(),
    #             attention_mask=attention_mask.cuda(),
    #             max_new_tokens=32,
    #             pad_token_id=self.tokenizer.eos_token_id,
    #             **gen_args,
    #         )

    #         generation_output = self.tokenizer.decode(generation_output[0][len(input_id[0]):]).replace("\n","").replace("{","").replace("}","").replace('"','').strip('".').replace(',,','').replace('original','').replace('Original','').split('rewritten')[0].split('revised')[0].replace('10','').split('.')[0]

    #     self.current_reward = None

    #     # update the total time steps
    #     self.current_data_idx += 1
    #     terminated = (self.steps >= self.args.max_steps)

    #     # remove the layer
    #     remove_icv_layers(self.model)

    #     return self._get_current_state(), self.current_reward, terminated, {"original_sentence": original_sentence, "icv": icv.detach().clone()}

    def _get_gen_args(self):
        if 'llama' in self.args.model_type:
            return {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [1642, 13492, 26036, 29908, self.tokenizer.encode('.10')[-1]]
            }
        elif 'falcon' in self.args.model_type:
            return {
                'do_sample': False,
                'num_beams': 10,
                'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166]
            }
        return {}

    def _finalize_chunk(self, chunk_indices, chunk_attn, original_hidden, generation_result, memory_elements):
        # Normalize attention weights
        chunk_attn = chunk_attn / (chunk_attn.sum() + 1e-9)
        
        # Calculate context-aware embeddings
        context_embed = (original_hidden * chunk_attn.unsqueeze(-1)).sum(dim=0)
        generated_embed = torch.stack([
            generation_result.hidden_states[-1][idx]
            for idx in chunk_indices
        ]).mean(dim=0)
        
        # Store chunk information
        memory_elements.append({
            "context_embed": context_embed.cpu(),
            "steering_vector": (generated_embed - context_embed).cpu(),
            "attention_pattern": chunk_attn.cpu(),
            "chunk_position": (chunk_indices[0], chunk_indices[-1])
        })


    def step(self, action):
        original_sentence, reference = self.raw_train_data[self.current_data_idx]['query'][0], self.raw_train_data[self.current_data_idx]['query'][1]

        self.current_state = self._get_hidden_state(original_sentence)
        
        icv = obtain_memory_icv(self.model, self.train_dataset.tokenize_each_demonstration(
            [(original_sentence, reference)], prefix=("", "")), rank=1
        )
        icv = icv[1:]
        
        # add the ICV to the current state
        add_icv_layers(self.model, torch.stack([icv],dim=1).cuda(), [self.args.lam])

        if 'llama' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [1642, 13492, 26036, 29908,self.tokenizer.encode('.10')[-1]]
            }
        elif 'falcon' in self.args.model_type:
            gen_args = {
                'do_sample': False,
                'num_beams': 10,
                'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166]
            }
        else:
            gen_args = {}
        
        with torch.no_grad():
            input_id = self.train_dataset[self.current_data_idx][0].unsqueeze(0) # ensure 2d input_ids
            attention_mask = self.train_dataset[self.current_data_idx][1].unsqueeze(0)
            generation_output = self.model.generate(
                input_ids=input_id.cuda(),
                attention_mask=attention_mask.cuda(),
                max_new_tokens=32,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_args,
            )

            generation_output = self.tokenizer.decode(generation_output[0][len(input_id[0]):]).replace("\n","").replace("{","").replace("}","").replace('"','').strip('".').replace(',,','').replace('original','').replace('Original','').split('rewritten')[0].split('revised')[0].replace('10','').split('.')[0]

        self.current_reward = None

        # update the total time steps
        self.current_data_idx += 1
        terminated = (self.steps >= self.args.max_steps)

        # remove the layer
        remove_icv_layers(self.model)

        return self._get_current_state(), self.current_reward, terminated, {"original_sentence": original_sentence, "icv": icv.detach().clone()}




class MemEnvEval(gymnasium.Env):
    def __init__(self, args, model_name, model, tokenizer, train_dataset, eval_dataset, raw_eval_data ,memory, global_icv):
        self.args = args
        self.model_name = model_name    
        self.model = model
        self.tokenizer = tokenizer

        self.model_shakespeare = RobertaForSequenceClassification.from_pretrained("/home/s223540177/ICV/evaluate/shakes/saves/shakespeare-classifier", num_labels=2).to("cuda")
        self.tokenizer_shakespeare = RobertaTokenizer.from_pretrained("/home/s223540177/ICV/evaluate/shakes/saves/shakespeare-classifier")


        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.raw_eval_data = raw_eval_data
        self.memory = memory

        print(len(eval_dataset))
        print(len(raw_eval_data))

        # build model and tokenizer
        model_to_hidden = {
            "llama-2": 4096,
            "falcon": 4544, 
        }
        
        model_to_n_layers = {
            "llama-2": 32,
            "falcon": 32,
        }

        self.n_layers = model_to_n_layers[self.model_name]
        self.hidden_size = model_to_hidden[self.model_name]

        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.meteor = evaluate.load('meteor')
        self.toxicity = OffensiveLanguageClassifier(custom_model_file='zoo:bot_adversarial_dialogue/multi_turn/model')
        

        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.n_layers, self.hidden_size),
            dtype=float
        )

        # state is discrete
        self.action_space = spaces.Discrete(1)

        self.global_icv = global_icv

        self.current_data_idx = 0
        self.current_reward = 0
        self.steps = 0

    def calculate_context_ratio(self, attentions, context_length):
        """
        Calculate the ratio of attention focused on the context tokens.
        
        Parameters:
            attentions (torch.Tensor): Attention scores, shape [batch_size, num_heads, seq_len, seq_len].
            context_length (int): Number of context tokens.

        Returns:
            torch.Tensor: Mean context ratio for all heads.
        """
        # Sum attention weights over context tokens
        context_attention = attentions[..., :context_length]  # Focus on context tokens
        context_sum = context_attention.sum(dim=-1)  # Sum over context tokens

        # Total attention per head
        total_attention = attentions.sum(dim=-1)

        # Avoid division by zero with epsilon
        epsilon = 1e-12
        context_ratio = context_sum / (total_attention + epsilon)

        # Average across all heads and batch
        avg_context_ratio = context_ratio.mean(dim=(0, 1))  # Average over heads and batch
        return avg_context_ratio


    def _get_train_dataset(self):
        return self.train_dataset

    def _get_obs(self):
        return {"previous_actions": self.previous_actions, "current_state": self.current_state, "current_ts": self.steps}

    def _get_current_state(self):
        return self.current_state

    def _get_hidden_state(self, inputs):
        text = [inputs]  # assuming the text is in 'text' field
        encoded_inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            last_hidden_state = self.model(**encoded_inputs, output_hidden_states=True).hidden_states[-1]
        
        weights_for_non_padding = encoded_inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

        return sentence_embeddings.cpu().detach().numpy()[0]


    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        # reset the total time steps
        self.steps = 0

        # reset the previous actions
        self.previous_actions = []

        # reset previous reward
        self.current_reward = 0

        original_sentence = self.raw_eval_data[self.current_data_idx]['query'][0]

        # get hidden state of the sentence
        self.current_state = self._get_hidden_state(original_sentence)

        state = self._get_current_state()

        # update the current training sample
        return state, {}

    def evaluate(self, metric_name, current_state, gold_label):
        if metric_name == "bleu":
            result = self.bleu.compute(predictions=[current_state], references=[gold_label])
        elif metric_name == "rouge":
            result = self.rouge.compute(predictions=[current_state], references=[gold_label])
        elif metric_name == "bertscore":
            result = self.bertscore.compute(predictions=[current_state], references=[gold_label], lang='en')
            result = round(np.mean(result['f1']), 4)
        elif metric_name == "meteor":
            result = self.meteor.compute(predictions=[current_state], references=[gold_label])
        elif metric_name == "toxicity":
            # special case where the result is a boolean
            ppred, prob = self.toxicity.contains_offensive_language(current_state)
            if prob > 0.9 and ppred:
                result = 0.0
            else:
                result = 1.0
        elif metric_name == "shakespeare":
            inputs = self.tokenizer_shakespeare(current_state, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            outputs = self.model_shakespeare(**inputs)
            prediction = outputs.logits.argmax(axis=-1).item()
            if prediction == 0:
                # this corresponds to modern style, give reward = 0
                result = 0.0
            else:
                result = 1.0
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        return result


    # def step(self, action):
    #     original_sentence = self.raw_eval_data[self.current_data_idx]['query'][0]
    #     reference = self.raw_eval_data[self.current_data_idx]['query'][1]

    #     self.current_state = self._get_hidden_state(original_sentence)

    #     icv, uncertainty = self.memory.read(self.args, self.current_state, distance_metric=self.args.distance_metric, k=self.args.n_neighbors)

    #     # # directly mix the global ICV with the current ICV
    #     if self.args.mix_icv:
    #         final_icv = (1 - uncertainty) * icv + uncertainty * self.global_icv
    #     else:
    #         final_icv = icv

    #     # add the ICV to the current state
    #     add_icv_layers(self.model, torch.stack([final_icv],dim=1).cuda(), [self.args.lam])
    
    #     if 'llama-2' in self.args.model_type:
    #         gen_args = {
    #             'temperature': 0.45,
    #             'do_sample': True,
    #             'top_k': 0,
    #             'top_p': 1.0,
    #             'eos_token_id': [1642, 13492, 26036, 29908,self.tokenizer.encode('.10')[-1], self.tokenizer.encode('\n')[-1]]
    #         }
    #     elif 'falcon' in self.args.model_type:
    #         gen_args = {
    #             'do_sample': False,
    #             'num_beams': 10,
    #             'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166]
    #         }
    #     else:
    #         gen_args = {}
        
    #     with torch.no_grad():
    #         input_id = self.eval_dataset[self.current_data_idx][0].unsqueeze(0) # ensure 2d input_ids
    #         attention_mask = self.eval_dataset[self.current_data_idx][1].unsqueeze(0)
    #         generation_output = self.model.generate(
    #             input_ids=input_id.cuda(),
    #             attention_mask=attention_mask.cuda(),
    #             max_new_tokens=32,
    #             pad_token_id=self.tokenizer.eos_token_id,
    #             **gen_args,
    #         )

    #         generation_output = self.tokenizer.decode(generation_output[0][len(input_id[0]):]).replace("\n","").replace("{","").replace("}","").replace('"','').strip('".').replace(',,','').replace('original','').replace('Original','').split('rewritten')[0].split('revised')[0].replace('10','').split('.')[0]

    #     # NOTE: Evaluation is done on the result file    
    #     self.current_reward = self.evaluate(self.args.metric, generation_output, reference)

    #     # update the total time steps
    #     self.current_data_idx += 1
    #     self.steps += 1

    #     terminated = True

    #     # remove the layer
    #     remove_icv_layers(self.model)

    #     return self._get_current_state(), self.current_reward, terminated, {"original_sentence": original_sentence, "icv": icv.detach().clone(), "generation": generation_output, "reference": reference}


    def step(self, action):
        original_sentence = self.raw_eval_data[self.current_data_idx]['query'][0]
        reference = self.raw_eval_data[self.current_data_idx]['query'][1]

        eval_data = self.eval_dataset[self.current_data_idx]
        self.current_state = self._get_hidden_state(original_sentence)

        current_sentence_with_ins = self.tokenizer.decode(eval_data[0], skip_special_tokens=True)

        if 'llama-2' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [1642, 13492, 26036, 29908,self.tokenizer.encode('.10')[-1], self.tokenizer.encode('\n')[-1]]
            }
        elif 'falcon' in self.args.model_type:
            gen_args = {
                'do_sample': False,
                'num_beams': 10,
                'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166, self.tokenizer.encode('\n')[-1]]
            }
        else:
            gen_args = {}
            
        cur_num_token = 0
        
        while cur_num_token < 32:
            icv, uncertainty = self.memory.read(self.args, self._get_hidden_state(original_sentence), original_sentence, distance_metric=self.args.distance_metric, k=self.args.n_neighbors)
            print("Uncertainty: ", uncertainty)
            if self.args.mix_strat == "hard":
                if uncertainty < self.args.mix_hard_threshold:
                    add_icv_layers(self.model, torch.stack([icv],dim=1).cuda(), [self.args.lam])
                else:
                    add_icv_layers(self.model, torch.stack([self.global_icv],dim=1).cuda(), [self.args.lam])
            elif self.args.mix_strat == "soft":
                if self.args.tune_weight > 0:
                    soft_icv = (1 - self.args.tune_weight) * icv + self.args.tune_weight * self.global_icv
                else:
                    soft_icv = (1 - uncertainty) * icv + uncertainty * self.global_icv
                add_icv_layers(self.model, torch.stack([soft_icv],dim=1).cuda(), [self.args.lam])
            else:
                raise ValueError(f"Unknown mix strategy: {self.args.mix_strat}")
            
            with torch.no_grad():
                # Tokenize input
                input_id = self.tokenizer(current_sentence_with_ins, return_tensors='pt')['input_ids']
                attention_mask = self.tokenizer(current_sentence_with_ins, return_tensors='pt')['attention_mask']
                
                # Generate only 1 new token per iteration
                generation_output = self.model.generate(
                    input_ids=input_id.cuda(),
                    attention_mask=attention_mask.cuda(),
                    max_new_tokens=1,  # Generate one token at a time
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_args,
                )
                
                # Extract the next token ID
                next_token_id = generation_output[0, -1].item()  # Last generated token
                
                # Stop if the next token is EOS
                if next_token_id in gen_args['eos_token_id']:
                    remove_icv_layers(self.model)
                    break
                
                # Append the next token to input_id
                input_id = torch.cat([input_id, torch.tensor([[next_token_id]])], dim=1)

                # get the next token
                current_sentence_with_ins = self.tokenizer.decode(input_id[0], skip_special_tokens=True)
                
                # Increment the token count
                cur_num_token += 1
            remove_icv_layers(self.model)

        

        # get the part to the right of Paraphrase, and get the text in between the first and last quotation marks
        generated_output = current_sentence_with_ins.split('Paraphrase')[1].split('"')[1]
        
        self.current_reward = None
        self.current_data_idx += 1


        return self._get_current_state(), self.current_reward, True, {"original_sentence": original_sentence, "icv": icv.detach().clone(), "generation": generated_output, "reference": reference}

