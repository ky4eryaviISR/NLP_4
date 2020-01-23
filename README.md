# NLP_4
## Train new model
<code>python extract_train.py Corpus.TRAIN.txt TRAIN.annotations model</code>

model - optional if not given default name for model file is model

## Predict on existed model
<code>python predict.py Corpus.TRAIN.txt Corpus.TRAIN.annotations feat_dict model</code>

feat_dict - file to convert features to numeric index
model - optional if not given default name for model file is model

## Evaluate the result to calculate precision, recall and F1-score
<code>python eval.py data/TRAIN.annotations output_annotation</code>
