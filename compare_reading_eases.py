from transformers import AutoModelForCausalLM, GPT2Tokenizer
from utils import models_dict, device
from metrics import monolingual_easy_language_german
import os
import pandas as pd
import torch

results_path = 'evaluation/'
results_file = results_path + 'reading_ease_comparison.csv'
if not os.path.exists(results_path):
    os.mkdir(results_path)
prompts = ["Das", "Heute", "Wir", "Die Türkei", "Dieses Haus", "Mein Vater"]
scores_cmp = pd.DataFrame(columns=["model", "strategy", "promt", "fre", "fkgl", "wiener", "avg_word_length", "avg_sentence_length", "words_per_sentence", "avg_syllables_per_word", "polysyllables", "output"])


def gen_from_prompts(encoded_prompt: torch.Tensor, model: AutoModelForCausalLM, strategy: str) -> torch.Tensor:
    """
    Use model.generate to generate some text based on the prompt. The prompt must already be encoded by the model's tokenizer.
    Three different pre-defined decoding strategies are supported:
    Set strategy='contrastive' to use contrastive search
    Set strategy='beam' to use beam search with 3 beams
    Set strategy='sampling' to use simple sampling

    """
    if strategy == 'contrastive':
        pred_ids = model.generate(encoded_prompt, max_new_tokens=100, top_k=4, penalty_alpha=0.6, repetition_penalty=1.4)
    elif strategy == 'beam':
        pred_ids = model.generate(encoded_prompt, max_new_tokens=100, num_beams=3, early_stopping=True,
                                  no_repeat_ngram_size=3)
    elif strategy == 'sampling':
        pred_ids = model.generate(encoded_prompt, max_new_tokens=100, top_k=1, top_p=0.92, do_sample=True, temperature=0.7,
                                  repetition_penalty=1.4)
    else:
        raise NotImplementedError
    return pred_ids


def eval_model(model:AutoModelForCausalLM, tokenizer: GPT2Tokenizer, prompts: list[str], model_name: str):
    """
    Evaluate a model on a list of string prompts
    """
    global scores_cmp
    for prompt in prompts:
        encoding = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
        for strat in ['contrastive', 'beam', 'sampling']:
            print('\t' + prompt)
            pred_ids = gen_from_prompts(encoding, model, strat)
            for i, pred_sents in enumerate(tokenizer.batch_decode(pred_ids, skip_special_tokens=True)):
                scores = monolingual_easy_language_german(pred_sents)
                scores['output'] = pred_sents
                scores['model'] = model_name
                scores['strategy'] = strat
                scores['promt'] = prompt

                scores_df = pd.DataFrame(scores, index=[0])
                scores_cmp = pd.concat([scores_cmp, scores_df], axis=0, ignore_index=True)


for model_name in models_dict.keys():
    print('Evaluating model', model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(models_dict[model_name])
    model = AutoModelForCausalLM.from_pretrained(models_dict[model_name]).to(device)
    eval_model(model, tokenizer, prompts, model_name)
    scores_cmp.to_csv(results_file, index=False)



