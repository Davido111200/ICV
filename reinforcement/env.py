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
import numpy as np
import evaluate
import torch
import random
from itertools import combinations
from tqdm import tqdm

from models import build_model, build_tokenizer
from common import mk_parser
from tasks import load_task

from utils.llm_layers import add_icv_layers, remove_icv_layers

from parlai.utils.safety import OffensiveLanguageClassifier

class ICVEnv(gymnasium.Env):
    def __init__(self, args, model_name, model, tokenizer, train_dataset, eval_dataset, icvs):
        self.args = args
        self.model_name = model_name    
        self.model = model
        self.tokenizer = tokenizer

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.icvs = icvs

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
            shape=(self.hidden_size,),
            dtype=float
        )

        # state is discrete
        self.action_space = spaces.Discrete(len(self.icvs) + 1)

        """
        The foloowing dictionary maps the actions from integers to embeddings
        The first action is to stop, this is a special case
        """
        self._action_to_icv = {0: -1}
        for i, icv in zip(range(len(self.icvs)), self.icvs):
            self._action_to_icv[i+1] = icv

        self.current_data_idx = 0

        self.current_reward = 0
        self.steps = 0
        
    def _get_train_dataset(self):
        return self.train_dataset

    def _get_obs(self):
        return {"previous_actions": self.previous_actions, "current_state": self.current_state, "current_ts": self.steps}

    def _get_current_state(self):
        return self.current_state

    def _get_actions(self):
        return self.actions

    def _get_hidden_state(self, inputs):
        # get the last hidden states of the sequence
        # Get model outputs
        # NOTE: this can be done if the RL model is the same as the model used to obtain the hidden states
        input_ids = inputs[0]
        attention_mask = inputs[1]
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0) # NOTE: this guarantees there will not be any shape error in generation
            attention_mask = attention_mask.unsqueeze(0) # NOTE: this guarantees there will not be any shape error in generation

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.to("cuda"), attention_mask=attention_mask.to("cuda"), output_hidden_states=True)
        
        # Access the hidden states
        hidden_states = outputs.hidden_states  # This will contain all hidden states from each layer

        # Get the last hidden state (from the last layer)
        last_hidden_state = hidden_states[-1]

        sentence_representation = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # convert to numpy
        sentence_representation = sentence_representation.cpu().numpy()

        self.current_state = sentence_representation

        return sentence_representation


    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        # reset the total time steps
        self.steps = 0

        # reset the previous actions
        self.previous_actions = []

        # reset previous reward
        self.current_reward = 0

        # get hidden state of the sentence
        self.current_state = self._get_hidden_state(self.train_dataset[self.current_data_idx])

        state = self._get_current_state()

        # update the current training sample
        self.current_data_idx += 1

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
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        return result

    def step(self, action):
        reference = self.train_dataset[self.current_data_idx]
        if action == 0 or self.steps >= self.args.max_steps:
            # action indicates stop
            return self._get_current_state(), self.current_reward, True, {}
        else:
            # map the action to get the ICV that we want to add to current state
            # NOTE: the current ICVs are not diverse enough. 52 different actions only result in 2 different generation.
            icv = self._action_to_icv[action]

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
            
            # get the current text state, and 
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
                print(f'generation: {generation_output}')

            # update reward    
            self.current_reward = self.evaluate(self.args.metric, generation_output, reference[2])

            # update previous actions
            self.previous_actions.append(action)

            # update the total time steps
            self.steps += 1

            terminated = (self.steps >= self.args.max_steps)

            # remove the layer
            remove_icv_layers(self.model)

            return self._get_current_state(), self.current_reward, terminated, {}




# if __name__ == "__main__":
#     parser = mk_parser()
#     args = parser.parse_args()
#     model_name = args.model_type
#     dataset_name = args.dataset

#     if model_name not in ["llama-2", "falcon"]:
#         raise ValueError(f"Unknown model name: {model_name}")

#     padding_side = 'right'

#     tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side=padding_side)
#     model = build_model(args.model_type, args.model_size, args.in_8bit)

#     TaskHandler = load_task(dataset_name)
#     task_agent = TaskHandler(args.prompt_version)
#     task_agent.set_seed(args.seed)
#     task_agent.do_load()

#     dataset = task_agent.mk_result_dataset(tokenizer, no_padding=True, prefix='Please paraphrase the following sentence.\n ')
    
#     if args.exemplar_method == "written":
#         exemplar_str = task_agent.handcrafted_exemplars()
#     elif args.exemplar_method == "random":
#         if args.test:
#             exemplar_str = task_agent.random_test_selected_exemplars(args.num_k_shots, prefix='Please paraphrase the following sentence.\n\n')
#         else:
#             exemplar_str = task_agent.random_selected_exemplars(args.num_k_shots, prefix='Please paraphrase the following sentence.\n\n')
#     elif args.exemplar_method == "stratified":
#         exemplar_str = task_agent.stratified_sampling(args.num_k_shots)
#     else:
#         raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

#     # print(task_agent.raw_data_sample)
#     all_data_sample = task_agent.raw_data_sample

#     # randomly select the pool of exemplars
#     # setting seed for random selection
#     random.seed(args.seed)
#     pool_exemplars = random.sample(all_data_sample, args.n_pool)

#     # NOTE: ICV is not influenced by the order of the examples, since finding the ICV relies only on the mean representation of the selected examples and their variance

#     # when random_selected_exemplars is executed, the task_agent._cached_ex_list is saved, thus is used for taking the icv
#     # the second input is in form [({'input_ids': [1644, 5122, 304, 416, 248, 12577, 204, 506, 12577, 2107, 240, 314, 3490, 875, 275, 9230, 2007, 5122], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})]
#     # with this in mind, we should encode the sentences prior to executing this function
#     icvs = []
#     elements = [_ for _ in range(args.n_pool)]
#     all_combinations = list(combinations(elements, 5))

#     # now we get all the ICVs by iteratively passing the examples (stack of args.num_k_shots) to find the icvs
#     pbar = tqdm(total=len(all_combinations))
#     for comb in all_combinations:
#         refined_comb = []
#         exemplars = [pool_exemplars[idx] for idx in comb]

#         for e in exemplars:
#             query = e["query"]
#             refined_comb.append(query)
    
#         icv, _ = task_agent.obtain_icv(
#             model, dataset.tokenize_each_demonstration(
#                 refined_comb, prefix=("", "")
#             ), rank=1
#         )
#         icvs.append(icv)
#         pbar.update(1)
    
#     env = ICVEnv(args, model_name, model, tokenizer, dataset, icvs)
    