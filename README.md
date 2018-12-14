nlpclass
==============================

Machine Translation Model for NLP class (NYU)

Evgenii Nikitin

William Godel

Installation instructions:
```
git clone https://github.com/crazyfrogspb/nlpclass.git
cd nlpclass
pip install -r requirements.txt
pip install -e .
```

Usage example:
```
python nlpclass/models/train_model.py vi --attention --bidirectional
```

Raw data folders should be put into data/raw directory in root.
