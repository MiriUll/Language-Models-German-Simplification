import torch
from tqdm import tqdm
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd
import textstat
from transformers import AutoModelForCausalLM, GPT2Tokenizer

textstat.set_lang("de")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Runinng on device: ", device)

models_dict = {
    'german_gpt': "tum-nlp/german-gpt2_easy",
    'german_gpt_orig': "dbmdz/german-gpt2",
    'wechsel': "tum-nlp/gpt2-wechsel-german_easy",
    'wechsel_orig': "benjamin/gpt2-wechsel-german",
    'gerpt2': "tum-nlp/gerpt2_easy",
    'gerpt2_orig': "benjamin/gerpt2",
    'oscar': "tum-nlp/gpt2-medium-german-finetune-oscar_easy",
    'oscar_orig': "ml6team/gpt2-medium-german-finetune-oscar",
    'mGPT': "tum-nlp/mGPT_easy",
    'mGPT_orig': "sberbank-ai/mGPT"
}

def modified_perplexity(model: AutoModelForCausalLM, encodings: list):
    max_length = model.config.n_positions
    stride = max_length

    nlls = []
    for sample in tqdm(encodings):
        sample_len = sample['input_ids'].size(1)
        sample_nlls = []
        for begin_loc in range(0, sample_len, stride):
            end_loc = min(begin_loc + stride, sample_len)
            # do not create samples with len 1
            if (sample_len - (begin_loc + stride)) == 1:
                #print(sample_len, begin_loc, end_loc)
                end_loc+=1
            input_ids = sample['input_ids'][:,begin_loc:end_loc].to(device)
            if len(input_ids[0])==1:
                print('producing nan')
            target_ids = input_ids.clone()
            #target_ids[:,:-1] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids).loss
            sample_nlls.append(outputs)

            if end_loc == sample_len:
                break
        #print(sample_len, len(sample_nlls))
        nlls.append(sum(sample_nlls) / len(sample_nlls))
    return torch.exp(torch.stack(nlls).sum() / len(nlls)).item()


def count_tokens(encoding):
    if type(encoding) == dict:
        encoding = encoding['input_ids']
    if type(encoding) == list:
        lens = [len(x['input_ids'][0]) for x in encoding]
        return sum(lens)
    encoding = encoding.reshape(-1)
    return encoding.shape(0)

def calculate_model_ppls_samplewise(model: AutoModelForCausalLM, tokenizer:GPT2Tokenizer, simple_texts: list[str], normal_texts: list[str])\
        -> (float, float, int, int):
    simple_encoding = [tokenizer(text, return_tensors="pt") for text in simple_texts]
    simple_ppl = modified_perplexity(model, simple_encoding)
    normal_encoding = [tokenizer(text, return_tensors="pt") for text in normal_texts]
    normal_ppl = modified_perplexity(model, normal_encoding)
    return simple_ppl, normal_ppl, count_tokens(simple_encoding), count_tokens(normal_encoding)


class ComplexityDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, regression):
#        dataframe['label'] = dataframe['label'].apply(lambda x: np.array(ast.literal_eval(x)))
        self.data = dataframe
        self.labels = list(dataframe.label.values)
#        self.labels = np.stack(dataframe.label.values)
        self.encodings = tokenizer(list(dataframe.text.values), truncation=True, max_length=1024, padding=True)
        self.regression = regression

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item = {'input_ids': torch.tensor(self.encodings['input_ids'][idx])}
        item['labels'] = float(self.labels[idx]) if self.regression else self.labels[idx]
        item['text'] = self.data.text[idx]
        return item

    def __len__(self):
        return len(self.data)

    def __get_labels__(self):
        return self.labels

    def __get_texts__(self):
        return self.data.texts.values

class AbstractDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, text_dataframe, stride_length, tokenizer, max_len):
        """
        text_dataframe: pandas dataframe with columns topic, phrase
        """
        assert ((text_dataframe.columns.values == ['topic', 'phrase']).all())
        self.texts = text_dataframe

        text_list = [unicodedata.normalize("NFC", s) for s in list(self.texts['phrase'].values)]

        self.stride_length = stride_length

        self.encodings = tokenizer(
            text_list,
            truncation=True,
            max_length=max_len,
            stride=stride_length,
            return_special_tokens_mask=True,
            return_overflowing_tokens=True,
            add_special_tokens=True
        )

    def get_source(self, idx) -> str:
        """
        Returns the source/topic of the requested item
        idx: index of a dataset item

        :return: str - the items original source
        """
        idx = self.encodings['overflow_to_sample_mapping'][idx]
        return self.get_name() + " -> " + self.texts.iloc[idx]['topic']

    def evaluate(self):
        """
        Evaluates the dataset on given metrics

        :return: pandas dataframe - summary of some metrics
        """

        # TODO replace by our metrics
        self.texts['fre'] = self.texts['phrase'].apply(lambda x: textstat.flesch_reading_ease(x))
        return self.texts.describe()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self) -> int:
        """
        Returns number of samples in data set

        :return: int - number of samples in data set
        """
        return len(self.encodings['input_ids'])

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the data set

        :return: str - name of the data set
        """
        pass

    @abstractmethod
    def get_columns(self) -> Iterable[str]:
        """
        Returns the names of all columns that the data set contains

        :return: list - names of the columns that are available
        """
        pass

class CombinedDataset(torch.utils.data.ConcatDataset):

    def __init__(self, datasets: Iterable[AbstractDataset]):
        super(CombinedDataset, self).__init__(datasets)

    def get_names(self) -> Iterable[str]:
        """
        Returns a list with the names of all data set that are contained in this combined data set

        :return: list - names of data sets in the data set collection
        """

        return [ds.get_name() for ds in self.datasets]

    def get_summary(self) -> str:
        total_items = 0
        individual_items = {}
        for dataset in self.datasets:
            individual_items[dataset.get_name()] = len(dataset)
            total_items += len(dataset)

        for key in individual_items.keys():
            individual_items[key] = "{:.2f}%".format((individual_items[key] / total_items) * 100)

        return f"Dataset contains {total_items} items {individual_items}"

class NewsData(AbstractDataset):
    def __init__(self, name, csv_file, stride_length, tokenizer, max_len):
        phrases = pd.read_csv(csv_file).fillna('text')
        texts = phrases.sort_values(['phrase_number']).groupby(['topic'])['phrase'].apply('\n'.join).reset_index()
        self.name = name
        super().__init__(texts, stride_length, tokenizer, max_len)

    def get_name(self) -> str:
        return self.name

    def get_columns(self) -> Iterable[str]:
        return self.texts.columns

def gen_and_eval(input, model, tokenizer):
    encoding = tokenizer(input, return_tensors="pt")['input_ids'].to(device)

    simple_texts = model.generate(encoding,
                                  repetition_penalty=1.4,
                                  max_length=64,
                                  top_k=0,
                                  temperature=0.7
                                  )

    # print("\nReading Ease: higher = better\n")

    reading_eases = []
    output_text = ''
    for text in tokenizer.batch_decode(simple_texts, skip_special_tokens=True):
        reading_ease = textstat.flesch_reading_ease(text)
        reading_eases.append(reading_ease)
        # print(f"Flesch Reading Ease: {reading_ease}\n")
        output_text += text
        # print(textwrap.fill(text, 130), '[...]')
    return reading_eases, output_text


def predict_text_proba(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return loss.item()