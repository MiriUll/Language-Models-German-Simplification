import pandas as pd
import torch.utils.data
from transformers import GPT2TokenizerFast, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import torch
from metrics import monolingual_easy_language_german
from transformers import TrainerCallback, TrainerState, TrainerControl
from tokenizers.processors import TemplateProcessing
from utils import device, NewsData, CombinedDataset, gen_and_eval, predict_text_proba, calculate_model_ppls_samplewise
from simctg.lossfunction import SimCTGLoss
from tokenizers import Tokenizer


fine_tune_models = [
    "dbmdz/german-gpt2",
    "benjamin/gerpt2",
    "benjamin/gpt2-wechsel-german",
    "ml6team/gpt2-medium-german-finetune-oscar",
    "sberbank-ai/mGPT"
]

PREFIX = "data/"
results_path = "results_"

class ComplexityCallback(TrainerCallback):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metrics = pd.DataFrame(
            columns=["steps", "fre", "fkgl", "wiener", "avg_word_length", "avg_sentence_length", "words_per_sentence",
                     "avg_syllables_per_word", "polysyllables", "text"])

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.on_evaluate(args=args, state=state, control=control, kwargs=kwargs)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('*Entered evaluation callback')
        #        encoding = torch.tensor([[self.tokenizer.bos_token_id]]).to(device)
        encoding = tokenizer("Dieses Haus ", return_tensors="pt")['input_ids'].to(device)
        #        pred_ids = model.generate(encoding, max_length=128, top_k=5, top_p=0.92, do_sample=True, temperature=0.7, num_return_sequences=3)
        pred_ids = model.generate(encoding, max_length=128, top_k=4, penalty_alpha=0.6, repetition_penalty=1.4)
        pred_sents = self.tokenizer.batch_decode(pred_ids)[0]
        scores = monolingual_easy_language_german(pred_sents)
        scores['steps'] = state.global_step
        scores['text'] = pred_sents
        self.metrics = self.metrics.append(scores, ignore_index=True)


margin = 0.5


class ContrastiveTrainer(Trainer):

    def __init__(self, tokenizer, **kwargs):
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get('input_ids')
        labels = torch.roll(inputs.get('labels'), 1)

        # forward computation
        bsz, seqlen = input_ids.size()
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        regular_loss = outputs.loss
        if self.label_smoother is not None:
            regular_loss = self.label_smoother(outputs, inputs.get("labels"), shift_labels=True)

        assert logits.size() == torch.Size([bsz, seqlen, model.config.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]

        # compute cl loss
        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        simctgloss = SimCTGLoss(margin=margin, vocab_size=self.vocab_size, pad_token_id=self.pad_token_id)
        cl_loss = simctgloss.contrastive_loss(cosine_scores, input_ids)

        simctg_loss = regular_loss + cl_loss
        return (simctg_loss, logits) if return_outputs else simctg_loss

for base_model_string in fine_tune_models:
    print("Finetuning", base_model_string)
    base_model_name = base_model_string.split('/')[-1]

    # the eos and bos tokens are defined
    bos = '<|bos|>'
    eos = '<|eos|>'
    pad = '<|pad|>'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}

    tokenizer_orig = AutoTokenizer.from_pretrained(base_model_string)
    tokenizer_orig.add_special_tokens(special_tokens_dict)
    tokenizer = Tokenizer.from_pretrained(base_model_string)
    tokenizer.post_processor = TemplateProcessing(
        single=bos + " $A " + eos,
        special_tokens=[(eos, tokenizer_orig.eos_token_id), (bos, tokenizer_orig.bos_token_id)],
    )
    tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    configuration = GPT2Config.from_pretrained(base_model_string, bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    #use_cache=False,
                                    )
    configuration.embd_pdrop = 0.1 #hyperparameter
    configuration.attn_pdrop = 0.1 #hyperparameter
    configuration.resid_pdrop = 0.1 #hyperparameter

    model = AutoModelForCausalLM.from_pretrained(base_model_string, config=configuration, force_download=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    stride_length = 64
    max_length = model.config.n_positions
    dataset_nachrichtenleicht = NewsData("NachrichtenLeicht", PREFIX + "nachrichtenleicht.csv", stride_length, tokenizer, max_length)
    dataset_ndr = NewsData("NDR", PREFIX + "ndr.csv", stride_length, tokenizer, max_length)
    dataset_einfachstars = NewsData("einfachstars", PREFIX + "einfachstars.csv", stride_length, tokenizer, max_length)
    dataset_hda = NewsData("hda", PREFIX + "hda_sprachtechnologie.csv", stride_length, tokenizer, max_length)
    dataset_lebenshilfe = NewsData("lebenshilfe", PREFIX + "lebenshilfe.csv", stride_length, tokenizer, max_length)
    dataset_hurraki = NewsData("hurraki", PREFIX + "hurraki.csv", stride_length, tokenizer, max_length)
    dataset_kurier = NewsData("kurier", PREFIX + "kurier.csv", stride_length, tokenizer, max_length)

    dataset_infoeasy = NewsData("Infoeasy", PREFIX + "Infoeasy.csv", stride_length, tokenizer, max_length)
    dataset_simple_german = NewsData("SimpleGerman", PREFIX + "simple_German_corpus.csv", stride_length, tokenizer, max_length)

    dataset = CombinedDataset([dataset_nachrichtenleicht,
                               dataset_hurraki,
                               dataset_ndr,
                               dataset_einfachstars,
                               dataset_hda,
                               dataset_lebenshilfe,
                               dataset_kurier,
                               dataset_infoeasy,
                               dataset_simple_german,
                               ])

    generator = torch.Generator()

    test_val_length = int(.1 * len(dataset))
    train_length = len(dataset) - test_val_length
    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, test_val_length],
                                                                 generator=generator)

    print(dataset.get_summary())
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_steps = 100

    # Finetuning for one epoch on all data
    training_args = TrainingArguments(
        num_train_epochs=1,
        output_dir=results_path+base_model_name,
        evaluation_strategy="steps",
        save_strategy='epoch',
        learning_rate=1e-4,  # hyperparamater
        weight_decay=0.01,  # hyperparamater
        #per_device_train_batch_size=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        #gradient_checkpointing=True,
        warmup_steps=200,
        logging_steps=eval_steps,
        eval_steps=eval_steps,
        #eval_accumulation_steps=1,
        fp16=True if device != 'cpu' else False,
        push_to_hub=True,
        hub_model_id=base_model_name+'_easy'
    )

    trainer = ContrastiveTrainer(
    #trainer = Trainer(
        tokenizer=tokenizer,
        model=model.to(device),
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
    )

    trainer.add_callback(ComplexityCallback(tokenizer))

    trainer.train()

    print("Saving tokenizer")
    tokenizer.save_pretrained(results_path+base_model_name)
    #trainer.push_to_hub()
    print("Saving complexity history")
    complexity_history = trainer.pop_callback(ComplexityCallback)
    complexity_history.metrics.to_csv(results_path + base_model_name + '/complexity.csv', index=False)

    model_orig = AutoModelForCausalLM.from_pretrained(base_model_string)
    tokenizer_orig = AutoTokenizer.from_pretrained(base_model_string)
    model_orig.to(device)

    with open(results_path + base_model_name + '/metrics.txt', 'w+') as outfile:
        outfile.write(f'Comparing: %s' %base_model_string)
        input = ["Die Türkei"]
        outfile.write("\nOriginal GPT")
        outfile.write(str(gen_and_eval(input, model_orig.eval(), tokenizer_orig)))
        outfile.write("\nFine-tuned GPT")
        outfile.write(str(gen_and_eval(input, model.eval(), tokenizer)))


        text_easy = "Leichte Sprache ist leichter zu lesen."
        text_complex = "Leichte Sprache ist eine speziell geregelte einfache Sprache."
        outfile.write("\n\nEasy text sample")
        outfile.write(f"\nOriginal GPT: {predict_text_proba(text_easy, model_orig.eval(), tokenizer_orig)}")
        outfile.write(f"\nFine-tuned GPT: {predict_text_proba(text_easy, model.eval(), tokenizer)}")
        outfile.write("\nComplex text sample")
        outfile.write(f"\nOriginal GPT: {predict_text_proba(text_complex, model_orig.eval(), tokenizer_orig)}")
        outfile.write(f"\nFine-tuned GPT {predict_text_proba(text_complex, model.eval(), tokenizer)}")

        outfile.write('\n\n Perplexity')
        mdr = pd.read_csv(PREFIX + "mdr_aligned_news.csv")
        simple_texts = mdr.dropna(subset=['simple_phrase'])['simple_phrase'].values.tolist()
        normal_texts = mdr.dropna(subset=['normal_phrase'])['normal_phrase'].values.tolist()
        simp_ppl, norm_ppl, _, _ = calculate_model_ppls_samplewise(model, tokenizer, simple_texts, normal_texts)
        outfile.write(f"\nPerplexity simple fine-tuned GPT: %f" %simp_ppl)
        outfile.write(f"\nPerplexity normal fine-tuned GPT: %f" %norm_ppl)
        simp_ppl_orig, norm_ppl_orig, _, _ = calculate_model_ppls_samplewise(model_orig, tokenizer_orig, simple_texts, normal_texts)
        outfile.write(f"\nPerplexity simple original GPT: %f" %simp_ppl_orig)
        outfile.write(f"\nPerplexity normal original GPT: %f" %norm_ppl_orig)