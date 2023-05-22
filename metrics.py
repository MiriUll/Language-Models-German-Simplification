from textstat import textstat
import numpy as np
from datasets import load_metric
from collections.abc import Iterable

textstat.set_lang("de")


# Requires "include_input_for_metrics = True" in the training_args
# use with lambda function:
# compute_metrics = lambda pred: compute_full_translation_metrics(input_tokenizer, output_tokenizer, pred)
# trainer = Trainer(...compute_metrics = compute_metrics)
def compute_full_translation_metrics(input_tokenizer, output_tokenizer, pred):
    monolingual_result = compute_monolingual_metrics(output_tokenizer, pred)
    translation_result = compute_translation_metrics(input_tokenizer, output_tokenizer, pred)
    return {**monolingual_result, **translation_result}


# Requires "include_input_for_metrics = True" in the training_args
# use with lambda function:
# compute_metrics = lambda pred: compute_translation_metrics(input_tokenizer, output_tokenizer, pred)
# trainer = Trainer(...compute_metrics = compute_metrics)
def compute_translation_metrics(input_tokenizer, output_tokenizer, pred):
    input_ids = pred.inputs
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    input_str = input_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    label_str = output_tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = output_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    pred_str_list = output_tokenizer.batch_decode(pred_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
    label_str_list = output_tokenizer.batch_decode(label_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
    pred_str_list = [pred.split() for pred in pred_str_list]
    label_str_list = [[label.split()] for label in label_str_list]

    sari = load_metric("sari")
    bleu = load_metric("bleu")

    translation_result = {
        **sari.compute(sources=input_str, predictions=pred_str, references=[[label] for label in label_str]),
        **bleu.compute(predictions=pred_str_list, references=label_str_list)
    }

    return {key: sum(value) / len(value) if isinstance(value, Iterable) else value for (key, value) in
            translation_result.items()}


# use with lambda function:
# compute_metrics = lambda pred: compute_monolingual_metrics(output_tokenizer, pred)
# trainer = Trainer(...compute_metrics = compute_metrics)
def compute_monolingual_metrics(tokenizer, pred):
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    result = monolingual_easy_language_german(pred_str)
    return result


def monolingual_easy_language_german(predictions):
    """
    Compute several metrics to evaluate readability of text samples

    :param predictions: Text sample(s)
    :return: dictionary with different scores
    """
    if not isinstance(predictions, str):
        return {
            # Designed for English, Adapted to German -> Higher = easier -> seems to work quite well
            "fre": np.array([textstat.flesch_reading_ease(pred) for pred in predictions]).mean(),
            # Designed for English -> Lower = easier
            "fkgl": np.array([textstat.flesch_kincaid_grade(pred) for pred in predictions]).mean(),
            # Designed for German (Lower = easier, see https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel)
            # There exist 4 different variants, randomly selected 1
            "wiener": np.array([textstat.wiener_sachtextformel(pred, 1) for pred in predictions]).mean(),
            # Some basic metrics that should always be reported as raw numbers as well
            "avg_word_length": np.array([textstat.avg_letter_per_word(pred) for pred in predictions]).mean(),
            "avg_sentence_length": np.array([textstat.avg_sentence_length(pred) for pred in predictions]).mean(),
            "words_per_sentence": np.array([textstat.words_per_sentence(pred) for pred in predictions]).mean(),
            "avg_syllables_per_word": np.array([textstat.avg_syllables_per_word(pred) for pred in predictions]).mean(),
            "polysyllables": np.array([textstat.polysyllabcount(pred) for pred in predictions]).mean()
        }
    else:
        return {
            # Designed for English, Adapted to German -> Higher = easier
            "fre": textstat.flesch_reading_ease(predictions),
            # Designed for English -> Lower = easier
            "fkgl": textstat.flesch_kincaid_grade(predictions),
            # Designed for German (Lower = easier, see https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel)
            "wiener": textstat.wiener_sachtextformel(predictions, 1),
            # There exist 4 different variants, randomly selected 1
            # Some basic metrics that should always be reported as raw numbers as well
            "avg_word_length": textstat.avg_letter_per_word(predictions),
            "avg_sentence_length": textstat.avg_sentence_length(predictions),
            "words_per_sentence": textstat.words_per_sentence(predictions),
            "avg_syllables_per_word": textstat.avg_syllables_per_word(predictions),
            "polysyllables": textstat.polysyllabcount(predictions)
        }
