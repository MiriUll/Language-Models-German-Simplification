from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from utils import models_dict, device, calculate_model_ppls_samplewise
import os

PREFIX = 'data/'

print('Loading dataset')
dataset_name = "mdr_aligned_news.csv"
assert os.path.isfile(PREFIX + dataset_name), "No MDR data found, please download it from our dataset page first"
mdr = pd.read_csv(PREFIX + dataset_name)
simple_texts = mdr.dropna(subset=['simple_phrase'])['simple_phrase'].values.tolist()
normal_texts = mdr.dropna(subset=['normal_phrase'])['normal_phrase'].values.tolist()

perplexity_cmp = []
for model_name in models_dict.keys():
    print('Evaluating model', model_name)
    tokenizer = AutoTokenizer.from_pretrained(models_dict[model_name])
    model = AutoModelForCausalLM.from_pretrained(models_dict[model_name]).to(device)
    simp_ppl, norm_ppl, simp_enc_len, norm_enc_len = calculate_model_ppls_samplewise(model, tokenizer, simple_texts, normal_texts)
    perplexity_cmp.append([model_name, simp_ppl, norm_ppl, simp_enc_len, norm_enc_len])
    print('Saving')
    perplexity_comp_df = pd.DataFrame(perplexity_cmp, columns=['model_name', 'simple_ppl', 'normal_ppl', 'num_tokens_simple', 'num_tokens_normal'])
    perplexity_comp_df.to_csv('evaluation/perplexity_comparison.csv', index=False)

