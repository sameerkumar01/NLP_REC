# NLP From Scratch

A course-style repository for learning major NLP algorithms by reading and running clear Python implementations. Core algorithms are written explicitly rather than hidden behind high-level helpers. Each topic will have a from-scratch implementation, a standard-library baseline, a comparison runner, a dataset download script, and a focused README.

> The GitHub repository is currently named `NLP_REC`; its root is the `nlp-from-scratch` learning project.

## Learning rules

- Core implementations use Python, NumPy, Pandas, and—only where useful for neural models—basic PyTorch tensor/autograd operations.
- No `CountVectorizer`, `TfidfVectorizer`, `Word2Vec`, `nn.LSTM`, `nn.Transformer`, `nn.MultiheadAttention`, `torchcrf`, `nltk.translate`, or `rouge-score` inside `from_scratch.py` files.
- Library implementations appear only in `baseline_library.py` files so the two approaches can be compared honestly.
- Every major equation is written beside the corresponding calculation in code.
- Large datasets are downloaded through scripts and small subsets are used by default for laptop/Colab-friendly runs.

## Course syllabus

| # | Topic | What it teaches |
|---:|---|---|
| 1 | [Text preprocessing pipeline](./01_text_preprocessing/) | Clean, tokenize, filter, stem, and lemmatize raw text. |
| 2 | [Bag of Words and N-grams](./02_bag_of_words_and_ngrams/) | Build vocabularies and unigram/bigram/trigram count matrices. |
| 3 | [TF-IDF and document retrieval](./03_tfidf_retrieval/) | Compute TF-IDF and cosine similarity without vectorizers. |
| 4 | [Edit distance and spelling correction](./04_edit_distance_spelling_correction/) | Use dynamic programming and a noisy-channel corrector. |
| 5 | [Statistical N-gram language model](./05_ngram_language_model/) | Estimate probabilities, smooth them, measure perplexity, and generate text. |
| 6 | [Naive Bayes text classifier](./06_naive_bayes_classifier/) | Classify text with manually computed multinomial likelihoods. |
| 7 | [Logistic regression for text](./07_logistic_regression_classifier/) | Train a binary classifier with explicit TF-IDF gradients. |
| 8 | [Linear SVM for text](./08_svm_classifier/) | Maximize the margin with hinge-loss gradient descent. |
| 9 | [Word2Vec](./09_word2vec/) | Train CBOW and Skip-gram embeddings with negative sampling. |
| 10 | [GloVe](./10_glove/) | Learn word vectors from a weighted co-occurrence objective. |
| 11 | [FastText-style subword embeddings](./11_fasttext_subword_embeddings/) | Represent words using character n-grams for OOV behavior. |
| 12 | [POS tagging with HMM + Viterbi](./12_pos_tagging_hmm_viterbi/) | Estimate transition/emission probabilities and decode tag sequences. |
| 13 | [Named Entity Recognition with a CRF](./13_ner_crf/) | Implement CRF scores, forward-backward, and Viterbi decoding. |
| 14 | [RNN, LSTM, and GRU cells](./14_rnn_lstm_gru/) | Build recurrent cells with explicit gate equations. |
| 15 | [Seq2Seq with attention](./15_seq2seq_attention/) | Connect encoder and decoder states with Bahdanau/Luong attention. |
| 16 | [Self-attention and Transformer](./16_transformer_from_scratch/) | Implement positional encoding, attention, normalization, and blocks. |
| 17 | [Byte Pair Encoding tokenizer](./17_bpe_tokenizer/) | Learn and apply the iterative BPE merge algorithm. |
| 18 | [Mini GPT-style language model](./18_mini_gpt/) | Combine tokenization, causal attention, and next-token training. |
| 19 | [Text-generation decoding strategies](./19_decoding_strategies/) | Compare greedy, beam, top-k, and nucleus sampling. |
| 20 | [BLEU and ROUGE metrics](./20_evaluation_metrics/) | Calculate overlap metrics without metric libraries. |
| 21 | [Extractive and abstractive summarization](./21_text_summarization/) | Use TextRank and the earlier neural models for summaries. |
| 22 | [LDA topic modeling](./22_lda_topic_modeling/) | Infer topics with collapsed Gibbs sampling. |
| 23 | [Optional pretrained Transformer capstone](./23_optional_pretrained_transformer/) | Compare a fine-tuned BERT-family model with the scratch implementations. |

## Current progress

- [x] Repository scaffold and syllabus
- [x] Topic 1: Text preprocessing pipeline
- [x] Topic 2: Bag of Words and N-grams
- [ ] Topics 3–23: implemented sequentially after review

## Working convention

Each topic is intentionally independent. Start in that topic's directory, read its README, download its dataset if needed, run `from_scratch.py`, then run `compare_results.py` to see where the transparent implementation differs from the library baseline.
