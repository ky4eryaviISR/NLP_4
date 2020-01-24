# NLP_4
## Train new model
<code>python extract_train.py Corpus.TRAIN.txt TRAIN.annotations model</code>

model - optional if not given default name for model file is model

## Predict on existed model
####MUST to execute previous file to create the model 
<code>python predict.py Corpus.TRAIN.txt Corpus.TRAIN.annotations feat_dict model</code>

feat_dict - optional file to convert features to numeric index
model - optional if not given default name for model file is model

## Evaluate the result to calculate precision, recall and F1-score
<code>python eval.py TRAIN.annotations output_annotation</code>
