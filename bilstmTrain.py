import sys
import spacy as spc
import dynet as dy
from collections import Counter
import random
import numpy as np
import codecs 
import util

REPR_WORDS = '1'
REPR_LSTM_OF_CHARS = '2'
REPR_PREFIX_SUFFIX = '3'
REPR_WORDS_N_LSTM_OF_CHARS = '4'
REPR_PRETRAINED_WORDS = '5'

class embedding(object):
    def __init__(self, model, vocabs, input_layer_size, option=3):
        self.m = model
        self.ils = input_layer_size
        self.repr = option
        self.vocabs = vocabs

        if option == '1':
            self.embeddings = self.m.add_lookup_parameters((self.vocabs.size(), self.ils))
            self.dict = self.vocabs.w2i
        if option == '2':
            self.builder = dy.LSTMBuilder(1, 50, input_layer_size, self.m)
            lstm = self.builder.initial_state()
            chars = set([c for w in self.vocabs.words for c in w])
            self.dict = { c:i for i,c in enumerate(chars) }
            self.embeddings = self.m.add_lookup_parameters((len(chars),50))
        if option == '3':
            words = self.vocabs.words
            longwords = [w for w in words if len(w) > 3]
            prefixes = [w[:3] for w in longwords] 
            suffixes = [w[-3:] for w in longwords]
            embed_size = len(words)+len(prefixes)+len(suffixes)
            self.word_dict = { w:i for i,w in enumerate(words) }
            self.prefix_dict = { w:i for i,w in enumerate(prefixes, len(self.word_dict)) }
            self.suffix_dict = { w:i for i,w in enumerate(suffixes, len(self.word_dict)+len(self.prefix_dict)) }
            self.embeddings = self.m.add_lookup_parameters((embed_size, self.ils)) 
        if option == '4':
            self.embeddings = self.m.add_lookup_parameters((self.vocabs.size(), self.ils))
            self.dict = self.vocabs.w2i
            self.builder = dy.LSTMBuilder(1, 50, self.ils, self.m)
            lstm = self.builder.initial_state()
            chars = set([c for w in self.vocabs.words for c in w])
            self.char_dict = { c:i for i,c in enumerate(chars) }
            self.char_embeddings = self.m.add_lookup_parameters((len(chars),50))
            self.W = self.m.add_parameters((self.ils, 2*self.ils))
        if option == '5':
            pass

    def __getitem__(self, word):
        if self.repr == REPR_WORDS:
            return self.embeddings[self.dict[word]]
        if self.repr == REPR_LSTM_OF_CHARS:
            lstm = self.builder.initial_state()
            outputs = lstm.transduce([self.embeddings[self.dict[c]] for c in word])
            return outputs[-1]
        if self.repr == REPR_PREFIX_SUFFIX:
            word_embed = pre_embed = suf_embed = None
            if word in self.word_dict:
                word_embed = self.embeddings[self.word_dict[word]]
            if len(word) <= 3:
                return word_embed
            pre = word[:3]
            suf = word[-3:]
            if pre in self.prefix_dict:
                pre_embed = self.embeddings[self.prefix_dict[word[:3]]]
            if suf in self.suffix_dict:
                suf_embed = self.embeddings[self.suffix_dict[word[-3:]]]
            new_embed = word_embed+pre_embed+suf_embed
            return word_embed + pre_embed + suf_embed
        if self.repr == REPR_WORDS_N_LSTM_OF_CHARS:
            word_embed = self.embeddings[self.dict[word]]
            lstm = self.builder.initial_state()
            outputs = lstm.transduce([self.char_embeddings[self.char_dict[c]] for c in word])
            word_n_char_embed = dy.concatenate([word_embed,outputs[-1]])
            return self.W * word_n_char_embed

def build_tagging_graph(words, real_tag, L1_builders, L2_builders, E):
    dy.renew_cg()
    L1_f_init, L1_b_init = [b1.initial_state() for b1 in L1_builders]

    word_embeddings = [E[w] for w in words]
    
    #if add_noise: word_embeddings = [dy.noise(we,0.1) for we in word_embeddings]
    L1_forward_sequences = L1_f_init.transduce(word_embeddings)
    L1_backwards_sequences = L1_b_init.transduce(reversed(word_embeddings))
    
    L2_f_init, L2_b_init = [b2.initial_state() for b2 in L2_builders]

    b_vectors = [dy.concatenate([L1_forward_output, L1_backwards_output]) for L1_forward_output, L1_backwards_output in zip(L1_forward_sequences, reversed(L1_backwards_sequences))] #L1_backwards_sequences)]

    L2_forward_sequences = L2_f_init.transduce(b_vectors)
    L2_backwards_sequences = L2_b_init.transduce(reversed(b_vectors))

    b_tag_vectors = [dy.concatenate([L2_forward_output, L2_backwards_output])
                 for L2_forward_output, L2_backwards_output
                 in zip(L2_forward_sequences, reversed(L2_backwards_sequences))]

    Net_O = Network_output
    errs = []

    b_tag_vector = dy.concatenate([L2_forward_sequences[-1],L2_backwards_sequences[0]])

    prediction = dy.softmax(Net_O*b_tag_vector)
    err = dy.pickneglogsoftmax(prediction, real_tag)
    errs.append(err)

    return np.argmax(prediction), dy.esum(errs)

def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-","(")
        sent = sent.replace("-RRB-",")")
        yield sent_id, sent

# changes the sentence representation
def representation(sentence_set,option=None):
    if option == 'NER':
        return [[w.ent_type if w.ent_type_ != '' else 'O' for w in s] for s in sentence_set]
    elif option == 'POS':
        return [[w.pos_ for w in s] for s in sentence_set]
    elif option == 'DEP':
        return [[w.dep_ for w in s] for s in sentence_set]
    elif option == 'TAG':
        return [[w.tag_ for w in s] for s in sentence_set]
    return [s.text.split() for s in sentence_set]

# adds the annotation from the annotation file
def add_tags(train_set, tags_file, true_tag):
    sentences = open(tags_file).read().split('\n')[:-1]

    train_set = [[s,False] for s in train_set]
    
    iterator = -1
    sentence_id = None
    for s in sentences:
        s = s.split('\t')[:3]
        if s[0] != sentence_id:
            iterator += 1
            sentence_id = s[0]
        if s[2] == true_tag:
            train_set[iterator][1] = True
    
    return train_set
    
epochs = 50
print_train_acc_every_x_sentences = 10000
print_dev_acc_every_x_sentences = 10000

L1_LSTM_layers = 1
L2_LSTM_layers = 1

L1_LSTM_input_dim = 50
L1_LSTM_hidden_dim = 240
L2_LSTM_input_dim = 2 * L1_LSTM_hidden_dim
L2_LSTM_hidden_dim = 240

if __name__ == '__main__':
    nlp = spc.load('en')
    
    train_file, annotation_file, dev_file, dev_annotation = sys.argv[1:]

    #initialize train_set
    train_set = []
    for sent_id, sent_str in read_lines(train_file):
        sent = nlp(sent_str)
        train_set.append(sent) #sent_str.encode("ascii")) #TODO correct this stuff

    #choose the way to represent the sentences - by default, just the words
    train_set = representation(train_set, 'TAG')

    train_set = add_tags(train_set, annotation_file, 'Live_In')

    print 'Train set size:', len(train_set)

    words=[]
    tags=[]
    words_counter = Counter()
    for sentence, tag in train_set:
        for word in sentence:
            words.append(word)
            words_counter[word]+=1
        tags.append(tag)
    words.append("_UNK_")

    vocab_words = util.Vocab.from_corpus([words])
    vocab_tags = util.Vocab.from_corpus([tags])
    UNK = vocab_words.w2i["_UNK_"]

#    i2t = { i:w for w,i in vocab_tags.w2i.items() }

    num_of_words = vocab_words.size()

    model = dy.ParameterCollection()
    trainer = dy.AdamTrainer(model)
    
    #TODO - use pretrained vectors
    E = embedding(model,vocab_words,L1_LSTM_input_dim, '1')

    L1_builders = [
        dy.LSTMBuilder(L1_LSTM_layers, L1_LSTM_input_dim, L1_LSTM_hidden_dim, model),
        dy.LSTMBuilder(L1_LSTM_layers, L1_LSTM_input_dim, L1_LSTM_hidden_dim, model),
    ]

    L2_builders = [
        dy.LSTMBuilder(L2_LSTM_layers, L2_LSTM_input_dim, L2_LSTM_hidden_dim, model),
        dy.LSTMBuilder(L2_LSTM_layers, L2_LSTM_input_dim, L2_LSTM_hidden_dim, model),
    ]

    for l in L1_builders:
        l.set_dropout(0.6)
    for l in L2_builders:
        l.set_dropout(0.6)
        
    Network_output = model.add_parameters((2, 2 * L2_LSTM_hidden_dim))

    
    tagged = loss = 0
    accuracy = 0.
    accuracy_list = []
    for epoch in range(epochs):
        good = 0.
        bad = 0.
        random.shuffle(train_set)
        for i, sentence in enumerate(train_set, 1):
            if i % 50 == 0:
                trainer.status()
                print 'Epoch:', epoch, 'Train set accuracy for last sentences: %.5f' % (good / (good+bad)), \
                    'Total sentences tagged:', i
                loss = 0
                tagged = 0
##            if i % print_dev_acc_every_x_sentences == 0:
##                accuracy = test_dev(dev_set, i)
##                accuracy_list.append(accuracy)
            ws = [word for word in sentence[0]]
            ps = vocab_tags.w2i[sentence[1]]

            predict, sum_errs = build_tagging_graph(ws, ps, L1_builders, L2_builders, E)

            if predict == ps:
                good+=1
            else:
                bad+=1
            squared = -sum_errs

            loss += sum_errs.value()
            tagged += 1

            sum_errs.backward()
            trainer.update()

    output_file = open('test.pred'+str(accuracy),'w')
    for sentence in test_set:
        tags = tag_sentence(sentence,L1_builders,L2_builders, test=True)
        for w,t in zip(sentence,tags):
            output_file.write(w[0]+'/'+t+' ')
        output_file.write('\n')

    if pos:
        tag_file = open('pos/tag','w')
        word_file = open('pos/words','w')
    else:
        tag_file = open('ner/tag','w')
        word_file = open('ner/words','w')
    print num_of_tags, "N'dd"
    print num_of_words, "hadpasa"
    tag_file.write(str(num_of_tags)+'\n')
    for w,i in vocab_words.w2i.items():
        word_file.write(w+'\t'+str(i)+'\n')
    for w,i in vocab_tags.w2i.items():
        tag_file.write(w+'\t'+str(i)+'\n')

    print E['_UNK_'].value()
    model.save(model_file+str(accuracy))
