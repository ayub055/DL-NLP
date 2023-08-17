def set_seed(seed = 42):
    '''
        For Reproducibility: Sets the seed of the entire notebook.
    '''

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Sets a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def split_indices(num_values, percentage):

    # Determine size of Validation set
    val_size = int(percentage * num_values)

    # Create random permutation of 0 to num_values-1
    idxs = np.random.permutation(num_values)
    return idxs[val_size:], idxs[:val_size]

def batch_fn(instn):

    sentence = torch.zeros(len(instn), 600)

    for i, v in enumerate(instn):
        sentence[i] = v[0]

    labels = torch.Tensor([x[1] for x in instn])

    return (sentence, labels)

def preprocess_ReviewsText(data, mode='positive'):
    
    nlp = spacy.load("en_core_web_md")
    data_set = []
    vocabulary = {}
    chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '.', ',']
    
    if mode == 'positive':
        sentiment = 1
    else :
        sentiment = 0
    
    for i, v in enumerate(data):

            # Removing Un-necessary symbols in our sentence
            for chars in chars_to_remove:
                v = v.replace(chars, " ", -1)

            sentence = torch.zeros(600)
            n = 0
            for token in nlp(v): 

                sentence[:300] += global_vectors.get_vecs_by_tokens(token.text, lower_case_backup=True)
                sentence[300:] += fasttext.get_vecs_by_tokens(token.text, lower_case_backup=True).squeeze(dim=0)
                n += 1

            # Taking mean
            sentence = sentence / n
            data_set.append((sentence, sentiment))
    
    return data_set     


def preprocess_tokenized_reviewText(pos, neg):
    
    nlp = spacy.load("en_core_web_md")
    data_set = []
    vocab = []
    chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '.', ',']
    tokenizer = get_tokenizer("basic_english")
    
    for line in pos:
        # Tokenizes the input text into words
        tokens = tokenizer(line)

        data_set.append((tokens, 1))
        # Adds the extracted words to a list
        vocab.extend(tokens)
    print(f"--- Positive Finished ---")
    
    for line in neg:
        # Tokenizes the input text into words
        tokens = tokenizer(line)

        data_set.append((tokens, 0))
        # Adds the extracted words to a list
        vocab.extend(tokens)
    print(f"--- Negative Finished ---")
    
    # Stores all the unique words in the dataset and their frequencies
    vocabulary = {}

    # Calculates the frequency of each unique word in the vocabulary
    for word in vocab:
        if word in vocabulary:
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

    print("Number of unique words in the vocabulary: ", len(vocabulary))
    return data_set, vocabulary

#-----------------------------------------------------------------------------------#

#len(set(vocab))
def sort_key(s):
    return len(s[0])

def assign_ids_to_words(vocab):
    # Stores the integer token for each unique word in the vocabulary
    ids_vocab = {}

    id = 0
    # Assigns words in the vocabulary to integer tokens
    for word, v in vocabulary.items():
        ids_vocab[word] = id
        id += 1
    
    return ids_vocab

# Tokenization function
def tokenize(corpus, ids_vocab):
    """
        Converts words in the dataset to integer tokens
    """

    tokenized_corpus = []
    for line, sentiment in corpus:
        new_line = []
        for i, word in enumerate(line):
            if word in ids_vocab and (i == 0 or word != line[i-1]):
                new_line.append(ids_vocab[word])

        new_line = torch.Tensor(new_line).long()
        tokenized_corpus.append((new_line, sentiment))

    return tokenized_corpus

def collate_fn_tokens(instn):

    sentence = [x[0] for x in instn]

    # Pre padding
    sen_len = [len(x[0]) for x in instn]
    max_len = max(sen_len)

    padded_sent = torch.zeros(1, max_len)
    sentence_pad = [torch.cat((torch.zeros(max_len-len(x[0])), x[0]), dim=0) for x in instn]
    
    for i in sentence_pad:
        padded_sent = torch.cat((padded_sent, i.unsqueeze(dim=0)), dim=0)
    padded_sent = padded_sent[1:].long()

    # Post padding
    #padded_sent = pad_sequence(sentence, batch_first=True, padding_value=0)

    labels = torch.Tensor([x[1] for x in instn])

    return (padded_sent, labels)

set_seed(1)
train_pos_indices, val_pos_indices = split_indices(len(positive_data), 0.1)
train_neg_indices, val_neg_indices = split_indices(len(negative_data), 0.1)

train_indices = np.concatenate((train_pos_indices, train_neg_indices+len(positive_data)-1))
val_indices = np.concatenate((val_pos_indices, val_neg_indices+len(positive_data)-1))
