import pandas as pd
from transformers import AutoTokenizer, GPT2ForSequenceClassification, \
    TrainingArguments, Trainer
from utils import models_dict, ComplexityDataset, device
from sklearn.metrics import mean_squared_error, accuracy_score
import json
import argparse
import os
import torch
seed = 42
#torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--regression', action='store_true')
args = parser.parse_args()

regression = args.regression
max_steps = [10, -1]

print('Loading dataset')
data_path = 'data/'
results_path = 'results_regression/' if regression else 'results_binary/'
if not os.path.exists(results_path):
    os.mkdir(results_path)
if regression:
    metric = mean_squared_error
    metric_name = 'mse'
    data_source = 'complexity'
else:
    metric = accuracy_score
    metric_name = 'accuracy'
    data_source = 'brandeins'
data_train = pd.read_csv(data_path + data_source + '_train.csv')
data_val = pd.read_csv(data_path + data_source + '_val.csv')
data_test = pd.read_csv(data_path + data_source + '_test.csv')

def compute_metric(eval_pred):
    predictions, labels = eval_pred
    if not regression:
        predictions = predictions.argmax(axis=1)
    metric_score = metric(labels, predictions)
    return {metric_name: metric_score}

def train_and_eval(orig, train, model_name, steps):
    eval0 = train.evaluate()
    eval0['step'] = 0
    eval0['epoch'] = 0.0
    print(eval0['eval_'+metric_name])
    train.train()
    log_path = f"{results_path}history_{model_name}_{steps}.json"
    history = train.state.log_history
    history.append(eval0)
    with open(log_path, 'w') as fout:
        json.dump(history, fout)

    print('Predicting and evaluation')
    y_pred = train.predict(test_data)
    return y_pred.metrics['test_'+metric_name]


metric_cmp = []
output_dir = 'trainer'
for model_name in models_dict.keys():
    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(models_dict[model_name])
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    train_data = ComplexityDataset(data_train, tokenizer, regression)
    val_data = ComplexityDataset(data_val, tokenizer, regression)
    test_data = ComplexityDataset(data_test, tokenizer, regression)
    for steps in max_steps:
        #output_dir = f"{results_path}results_{model_name}_{steps}"
        #if os.path.exists(output_dir):
        #    print(f"Results for {model_name} and {steps} steps exists. -> Skipping")
        #    continue
        print(f"Loading models for {steps} step training")
        def model_init(model_path):
            if regression:
                model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device)
            else:
                model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=2,
                                                                     problem_type="single_label_classification").to(device)
            model.config.pad_token_id = tokenizer.pad_token_id
            #for param in model.base_model.parameters():
            #    param.requires_grad = False
#            if model.config.n_embd == 768:
#                model.score.weight = torch.load('data/regression_weight_init.pt') if regression \
#                    else torch.load('data/binary_weight_init.pt')
            return model
        def my_model_init():
            return model_init(models_dict[model_name])

        #model = my_model_init()
        trainings_args = TrainingArguments(output_dir=output_dir, auto_find_batch_size=True,
                                           max_steps=steps, num_train_epochs=1,
                                           evaluation_strategy="steps", eval_steps=1, 
                                           seed=seed, data_seed=seed,
                                           #save_strategy='epoch')
                                           #save_steps=steps
                                           )
        print(f"Training {model_name} for {steps} training steps")
        trainer = Trainer(args=trainings_args, train_dataset=train_data, eval_dataset=val_data,
                          compute_metrics=compute_metric, 
                          model_init=my_model_init)
                          #model=model)
        metric_res = train_and_eval(False, trainer, model_name, steps)
        metric_cmp.append([model_name, steps, metric_res])

        print('Saving')
        metric_cmp_df = pd.DataFrame(metric_cmp, columns=['model_name', 'num_training_steps', metric_name])
        csv_name = results_path + 'downstream_comparison.csv'
        metric_cmp_df.to_csv(csv_name, index=False)
        del trainer
#        del model
