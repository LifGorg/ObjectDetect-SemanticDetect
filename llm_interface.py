from transformers import GPT2Tokenizer, AutoModelForCausalLM, pipeline, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, XLMTokenizer, XLMWithLMHeadModel
import json
import torch
from torch.nn import functional as F
import numpy as np
import pickle
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from sinkhorn_knopp import sinkhorn_knopp as skp

CUDA_DEVICE = 7

class Node:
    def __init__(self, value):
        self.value = value #either "string" or token
        self.children = defaultdict(int)
        self.token_children = []
        self.score = 0

class LLM_Interface:
    def __init__(self, model_path, objects, prompt, geo, llm_type="causal", affin_type="not pair", targets=[[], []], name=''):
        if llm_type == "causal":
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path) 
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.generator = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, device=CUDA_DEVICE)
        elif llm_type == "mlm":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            self.model = RobertaForMaskedLM.from_pretrained("roberta-large")
            self.generator = pipeline(task="fill-mask", model=self.model, tokenizer=self.tokenizer, device=CUDA_DEVICE) #check gpu usage
        elif llm_type == "xlm":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            self.model = RobertaForMaskedLM.from_pretrained("roberta-large")
            self.generator = pipeline(task="fill-mask", model=self.model, tokenizer=self.tokenizer, device=CUDA_DEVICE) #check gpu usage
        self.llm_type = llm_type
        self.objects, self.objects_dict, self.o = objects, None, len(objects)
        self.geo, self.g = geo, len(geo)
        self.prompt = prompt
        self.targets = targets
        self.name = name

        self.func = torch.nn.functional.log_softmax
        self.func_softmax = torch.nn.functional.softmax
        if affin_type == "pair":
            self.affinity_matrix = self.get_affinity_scores_target_pair_phrase()
        else:
            self.affinity_matrix = self.get_affinity_scores_normalize_by_prior()
        with open(self.name,'wb') as f:
            pickle.dump(self.affinity_matrix, f)
        # print(self.affinity_matrix)
        # print(self.affinity_matrix.shape)

    def create_object_dict(self, objects, prompt):
        offset = 1 if self.llm_type=="mlm" else 0 #due to the nature of the tokenizer
        self.objects_dict = {}
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_size = prompt_tokens.size(dim=1) - 1
        for i in objects:
            input_string = prompt + i + "."
            print(input_string)
            print(self.tokenizer(input_string, return_tensors="pt").input_ids)
            object_token = self.tokenizer(input_string, return_tensors="pt").input_ids[:, prompt_size-offset:].to(CUDA_DEVICE)
            # print(object_token)
            curr_token = torch.tensor(-1).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
            moded_t = torch.cat((curr_token, object_token), dim=1)
            self.objects_dict[i] = moded_t
        # print(self.objects_dict)
        return self.objects_dict
    
    def create_graph(self, objects, inputs):
        objects_dict = self.create_object_dict(objects, inputs)
        graph = Node(-1)
        for obj in range(self.o):
            tokens = objects_dict[self.objects[obj]][0]
            curr = graph
            for i in range(1, tokens.size(dim=0)+1):
                if i == tokens.size(dim=0):
                    new = Node(self.objects[obj])
                    curr.children[self.objects[obj]] = new
                    curr.token_children.append(new.value)
                    continue

                curr_token = tokens[i].item()
                if curr_token not in curr.token_children:
                    new = Node(curr_token)
                    curr.children[curr_token] = new
                    curr.token_children.append(curr_token)
                    curr = new
                else:
                    curr = curr.children[curr_token]

        return graph
    
    #condition on a set of objects, create a bunch of scenes, recondition the probabilities 

    def generate(self):
        # print("Ready!\n")
        # #INPUT TEST
        # while prompt != "q":
        #     inputs = tokenizer(prompt, return_tensors="pt")
        #     print(inputs)
        #     inputs.to(CUDA_DEVICE)
        #     generate_ids = model.generate(inputs.input_ids, max_length=60)
        #     outputs = model(inputs.input_ids)t to 
        #     #model(prompt)

        #     print(outputs.logits.shape) #see what is outputs.attention
        #     testoutput = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #     print(testoutput)
        return None

    def dfs_update_node(self, graph, mod_prompt):
        # print(mod_prompt)
        if mod_prompt.device == "cpu":
            mod_prompt.to(CUDA_DEVICE)
        # print(self.model(mod_prompt).logits.shape)
        if self.llm_type == "causal":
            last_token_vocab = self.func(self.model(mod_prompt).logits[:, -1, :])
        if self.llm_type == "mlm":
            log_s_output = self.func(self.model(mod_prompt).logits, dim=-1)
            mask_token_index = (mod_prompt == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
        # val = self.func_softmax(last_token_vocab)
        # print(last_token_vocab.shape, last_token_vocab, torch.sum(val), torch.max(val))
        moded_prompt = mod_prompt
        for vertex in graph.token_children:
            if isinstance(vertex, int):
                if self.llm_type == "causal":
                    last_token_score = last_token_vocab[:, vertex]
                elif self.llm_type == "mlm":
                    last_token_score = log_s_output[0, mask_token_index, vertex]
                
                graph.children[vertex].score = last_token_score.item()
                curr_token = torch.tensor(vertex).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
                if self.llm_type == "causal":
                    moded_prompt = torch.cat((mod_prompt, curr_token), dim=1)
                elif self.llm_type == "mlm":
                    mask_tok = torch.tensor(self.tokenizer.mask_token_id).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
                    moded_prompt = torch.cat((mod_prompt[:, :mask_token_index].to(CUDA_DEVICE), curr_token, mask_tok, mod_prompt[:, mask_token_index+1:].to(CUDA_DEVICE)), dim=1)
                self.dfs_update_node(graph.children[vertex], moded_prompt)
    
    def dfs(self, graph, prob):
        for i in graph.token_children:
            vertex = graph.children[i]
            if isinstance(i, int):
                self.dfs(vertex, prob+graph.score)
            else:
                self.dict_scores[i] = prob+graph.score
    
    def visualize_graph(self, graph):
        # print(graph.value, graph.score, graph.token_children)
        for i in graph.token_children:
            vertex = graph.children[i]
            self.visualize_graph(vertex)

    def update_graph(self, graph, mod_prompt):
        offset = 1 if self.llm_type == "causal" else 0
        self.dict_scores = defaultdict(float)
        mod_prompt.to(CUDA_DEVICE)
        mm_prompt = mod_prompt.input_ids
        
        # print(mod_prompt) 
        # print(self.dict_scores, mm_prompt)
        # print("input", mm_prompt[:, :mm_prompt.size(dim=1)])
        self.dfs_update_node(graph, mm_prompt[:, :mm_prompt.size(dim=1)-offset]) #Node=-1 is a fake node
        self.visualize_graph(graph)
        self.dfs(graph, 0)

        # self.average_log_prob(self.dict_scores) Average the log the probabilities, didn't really work as well, maybe more creative

        # denom = np.logaddexp.reduce(np.array(list(self.dict_scores.values())))
        # for obj in range(self.o):
        #     self.dict_scores[self.objects[obj]] -= denom
            
        return self.dict_scores
    
    def average_log_prob(self, dict_scores):
        for d in dict_scores:
            dict_scores[d] = dict_scores[d] / (self.objects_dict[d].size(dim=1)-1)
        

    def get_affinity_scores(self):
        #tokenize the wordinput_ids
        affinity_matrix = np.zeros((self.g, self.o, self.o))

        for g in range(self.g):
            for i in range(self.o):
                current_obj, curr_geo = self.objects[i], self.geo[g]
                input_string = self.prompt.format(current_obj, curr_geo)
                dict_scores = {t: None for t in self.objects}                
                graph = self.create_graph(self.objects, input_string)
                input_string = input_string + "<mask>." if self.llm_type == "mlm" else input_string
                inputs = self.tokenizer(input_string, return_tensors="pt")
                # self.visualize_graph(graph)
                #TODO look into weighting the current_obj word more than the rest of the words in the sentence
                #TODO take average across multiple tokens after suming the log prob for the multiple tokens, beam search doesn't work on hot glue gun need current apporach for that but current approach doesn't work salad fork where salad token brings down the word down even if the word is fork                 
                dict_scores = self.update_graph(graph, inputs)

                denom = np.logaddexp.reduce(np.array(list(self.dict_scores.values())))
                for obj in range(self.o):
                    self.dict_scores[self.objects[obj]] -= denom

                for j in range(self.o):
                    affinity_matrix[g, i, j] = dict_scores[self.objects[j]]
                        
        return affinity_matrix


    def get_affinity_scores_normalize_by_prior(self):
        #tokenize the wordinput_ids
        affinity_matrix = np.zeros((self.g, self.o, self.o))
        p_string = "A shelf can contain objects such as "
        input_string = p_string
        prior_dict_scores = {t: None for t in self.objects}
        graph = self.create_graph(self.objects, input_string)
        input_string = input_string + "<mask>." if self.llm_type == "mlm" else input_string
        inputs = self.tokenizer(input_string, return_tensors="pt")
        prior_dict_scores = self.update_graph(graph, inputs)

        for g in range(self.g):
            for i in range(self.o):
                current_obj, curr_geo = self.objects[i], self.geo[g]
                input_string = self.prompt.format(current_obj, curr_geo)
                dict_scores = {t: None for t in self.objects}                
                graph = self.create_graph(self.objects, input_string)
                input_string = input_string + "<mask>." if self.llm_type == "mlm" else input_string
                inputs = self.tokenizer(input_string, return_tensors="pt")
                dict_scores = self.update_graph(graph, inputs)
                
                print("dict_scores", dict_scores)
                print("prior", prior_dict_scores)

                for j in range(self.o):
                    dict_scores[self.objects[j]] = dict_scores[self.objects[j]] #- prior_dict_scores[self.objects[j]]
                
                print("post norm dict_scores", dict_scores)

                # denom = np.logaddexp.reduce(np.array(list(dict_scores.values())))
                # print(denom)
                # for obj in range(self.o):
                #     dict_scores[self.objects[obj]] -= denom
                print("post_denom", dict_scores)
                print("sum to one", np.sum(np.exp(np.array(list(dict_scores.values())))))
                for j in range(self.o):
                    affinity_matrix[g, i, j] = dict_scores[self.objects[j]]
        
        # denom = np.logaddexp.reduce(np.array(list(dict_scores.values())))
                # print(denom)
                # for obj in range(self.o):
                #     dict_scores[self.objects[obj]] -= denom

        # meaned_matrix = np.divide(np.sum(affinity_matrix.transpose(0, 2, 1), affinity_matrix), 2)
        print(affinity_matrix)
        sk = skp.SinkhornKnopp()
        affint_ds = np.expand_dims(sk.fit(np.exp(affinity_matrix).squeeze()), 0)
        return affinity_matrix
        # print(affint_ds)
        # return np.exp(affinity_matrix)


    def get_affinity_scores_target_pair_phrase(self):
        #tokenize the wordinput_ids

        #PAIR TARGET APPROACH
        # prompt = "In a shelf, does the {} go {} or {} the {}? The {} goes {} the {}."
        # prompt = "In a shelf, the {} goes {} the {}."
        # targets = [['close to'], ['far away from']]
        # model = LLM_Interface(path, objects, prompt, geo, "mlm", "pair", targets)
        
        affinity_matrix = np.zeros((self.o, self.o))

        for i in range(self.o):
            for j in range(self.o):
                for first_option in self.targets[0]:
                    for second_option in self.targets[1]:
                        input_masked_string = self.prompt.format(self.objects[i], first_option, second_option, self.objects[j], self.objects[i], "<mask>", self.objects[j])
                        input_masked_tokens = self.tokenizer(input_masked_string, return_tensors="pt").input_ids #tokens of the input masked string
                        input_first_opt_string = self.prompt.format(self.objects[i], first_option, second_option, self.objects[j], self.objects[i], first_option, self.objects[j])
                        input_first_opt_tokens = self.tokenizer(input_first_opt_string, return_tensors="pt").input_ids #tokens of the input first option string
                        input_second_opt_string = self.prompt.format(self.objects[i], first_option, second_option, self.objects[j], self.objects[i], second_option, self.objects[j])
                        input_second_opt_tokens = self.tokenizer(input_second_opt_string, return_tensors="pt").input_ids #tokens of the input second option string
                        # input_masked_string = self.prompt.format(self.objects[i], "<mask>", self.objects[j])
                        # input_masked_tokens = self.tokenizer(input_masked_string, return_tensors="pt").input_ids #tokens of the input masked string
                        # input_first_opt_string = self.prompt.format(self.objects[i], first_option, self.objects[j])
                        # input_first_opt_tokens = self.tokenizer(input_first_opt_string, return_tensors="pt").input_ids #tokens of the input first option string
                        # input_second_opt_string = self.prompt.format(self.objects[i], second_option, self.objects[j])
                        # input_second_opt_tokens = self.tokenizer(input_second_opt_string, return_tensors="pt").input_ids #tokens of the input second option string
                        print(input_first_opt_string)
                        print(input_second_opt_string)
                        tokens_first_option = []
                        tokens_second_option = []
                        count, asd = 0, 0
                        while count < input_first_opt_tokens.size(dim=1):
                            if input_first_opt_tokens[0][count].item() == input_masked_tokens[0][asd].item():
                                asd += 1
                                count += 1
                                continue
                            else:
                                if self.tokenizer.mask_token_id == input_masked_tokens[0][asd].item():
                                    asd += 1
                                tokens_first_option.append(input_first_opt_tokens[0][count].item())
                            count += 1
                        count, asd = 0, 0
                        while count < input_second_opt_tokens.size(dim=1):
                            if input_second_opt_tokens[0][count].item() == input_masked_tokens[0][asd].item():
                                asd += 1
                                count += 1
                                continue
                            else:
                                if self.tokenizer.mask_token_id == input_masked_tokens[0][asd].item():
                                    asd += 1
                                tokens_second_option.append(input_second_opt_tokens[0][count].item())
                            count += 1
                        first_log_prob = 0
                        second_log_prob = 0
                        fcount, scount = 0, 0
                        mod_prompt = input_masked_tokens.to(CUDA_DEVICE)
                        while fcount < len(tokens_first_option):
                            log_s_output = self.func(self.model(mod_prompt).logits, dim=-1)
                            mask_token_index = (mod_prompt == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                            curr = tokens_first_option[fcount]
                            token_score = log_s_output[0, mask_token_index, curr].item()
                            first_log_prob += token_score
                            curr_token = torch.tensor(curr).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
                            mask_tok = torch.tensor(self.tokenizer.mask_token_id).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
                            moded_prompt = torch.cat((mod_prompt[:, :mask_token_index].to(CUDA_DEVICE), curr_token, mask_tok, mod_prompt[:, mask_token_index+1:].to(CUDA_DEVICE)), dim=1)
                            mod_prompt = moded_prompt
                            fcount+= 1

                        mod_prompt = input_masked_tokens.to(CUDA_DEVICE)
                        while scount < len(tokens_second_option):
                            log_s_output = self.func(self.model(mod_prompt).logits, dim=-1)
                            mask_token_index = (mod_prompt == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                            curr = tokens_second_option[scount]
                            token_score = log_s_output[0, mask_token_index, curr].item()
                            second_log_prob += token_score
                            curr_token = torch.tensor(curr).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
                            mask_tok = torch.tensor(self.tokenizer.mask_token_id).unsqueeze(0).unsqueeze(0).to(CUDA_DEVICE)
                            moded_prompt = torch.cat((mod_prompt[:, :mask_token_index].to(CUDA_DEVICE), curr_token, mask_tok, mod_prompt[:, mask_token_index+1:].to(CUDA_DEVICE)), dim=1)
                            mod_prompt = moded_prompt
                            scount+= 1
                        
                        print(self.objects[i], self.objects[j], first_log_prob, second_log_prob, first_log_prob - np.logaddexp.reduce(np.array([first_log_prob, second_log_prob])))
                        affinity_matrix[i, j] = first_log_prob - np.logaddexp.reduce(np.array([first_log_prob, second_log_prob]))
        
        return affinity_matrix

def visualize_affinity_matrix(affinity_matrix, object_type_list, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(affinity_matrix, cmap='hot', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(object_type_list)))
    ax.set_yticks(np.arange(len(object_type_list)))
    ax.set_xticklabels(object_type_list, rotation='vertical')
    ax.set_yticklabels(object_type_list)
    for label in ax.xaxis.get_ticklabels():
        label.set_visible(True)
    for label in ax.yaxis.get_ticklabels():
        label.set_visible(True)
    fig.colorbar(im)
    plt.savefig(name)
