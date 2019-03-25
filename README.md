# Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
Tensorflow implementation of [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)
### Environment
- python 3.5
- tensorflow 1.2.1
- nltk
- codecs

### Run
- download [SemEval2010_task8_all_data.zip](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50), uncompress
- move "./utils/make_format_data.ipynb" to dataset folder and run
- move "test.txt train.txt valid.txt" in dataset folder to "./data"
- build vocabulary
```bash
python make_vocab.py
```
- download [Glove.6B](http://nlp.stanford.edu/data/glove.6B.zip) and build w2v_matrix
```bash
python make_w2v.py --embedding_path=./glove.6B/glove.6B.100d.txt --vocab_path=./data/word_vocab --vocab_size=10000000
```
- run run_nre.py
```bash
python run_nre.py
``` 

### Results
With my own valid set, i only get macro-f1(OFFICIAL SCORE) 80.31.
### Reference
[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)
