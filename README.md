# Language-Models-German-Simplification
This repository contains the code for the paper "Language Models for German Text Simplification: Overcoming Parallel Data Scarcity through Style-specific Pre-training".  
You can find the published models in the [Huggingface hub](https://huggingface.co/tum-nlp).  
The data used for this project can be downloaded using our [scrapers](https://github.com/brjezierski/scrapers).

## Fine-tuning language models
Use the [``finetuning.py``](https://github.com/MiriUll/Language-Models-German-Simplification/blob/main/finetuning.py) script to create your own Leichte Sprache language models. You need to download/scrape the monolingual corpus from [here](https://github.com/brjezierski/scrapers) first.

## Re-creating the results from the paper
The evaluations for the perplexity scores, the readability of the language model outouts and the downstream task performance are provided in the respective scripts. We also publish the answers from the human grammar evaluation in the file [``evaluation/Evaluierung von large language models.csv``](https://github.com/MiriUll/Language-Models-German-Simplification/blob/main/evaluation/Evaluierung%20von%20large%20language%20models.csv). You can analyze these results with the [human evaluation notebook](https://github.com/MiriUll/Language-Models-German-Simplification/blob/main/human_eval.ipynb).   

For the application of the language models as ATS decoders, please refer to the original [Github repo](https://github.com/a-rios/longmbart). You can find the fine-tuned simplification model on [Huggingface](https://huggingface.co/josh-oo/custom-decoder-ats). The simplification results are stored in the original [tensorboard_logs](https://github.com/MiriUll/Language-Models-German-Simplification/tree/main/evaluation/tensorboard_logs_simplification).

## Citation
If you use our models or the code in one of our repos, please use the following citation:  
```bibtex
@inproceedings{anschutz-etal-2023-language,  
    title = "Language Models for {G}erman Text Simplification: Overcoming Parallel Data Scarcity through Style-specific Pre-training",  
    author = {Ansch{\"u}tz, Miriam  and Oehms, Joshua  and Wimmer, Thomas  and Jezierski, Bart{\l}omiej  and Groh, Georg},  
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",  
    month = jul,  
    year = "2023",  
    address = "Toronto, Canada",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2023.findings-acl.74",  
    pages = "1147--1158",  
}
```
