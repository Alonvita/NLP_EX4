import sys
import spacy as spc
import dynet as dy
from collections import Counter
import random
import numpy as np
import rule_based as rb
import codecs 
import util

def read_lines(fname, is_annotated):
    print fname, "fname"
    if is_annotated:
        for line in codecs.open(fname, encoding="utf8"):
            if line == '': continue
            sent_id = line.split('\t')[0][4:]
            annotation = line.split('(')[0].split('\t')[2]
            sent = line.split("(")[1][:-1]
            sent = sent.replace("-LRB-","(")
            sent = sent.replace("-RRB-",")")
            yield sent_id, annotation, sent
    else:     
        for line in codecs.open(fname, encoding="utf8"):
            if line == '': continue
            sent_id = line.split('\t')[0][4:]

            sent = line.split("(")[1][:-1]
            sent = sent.replace("-LRB-","(")
            sent = sent.replace("-RRB-",")")
            yield sent_id, sent

# changes the sentence representation
def representation(sentence_set,option=None):
    if option == 'NER':
        return [[w.ent_type_ if w.ent_type_ != '' else 'O' for w in s] for s in sentence_set]
    elif option == 'POS':
        return [[w.pos_ for w in s] for s in sentence_set]
    elif option == 'DEP':
        return [[w.dep_ for w in s] for s in sentence_set]
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

def create_file(src_file, output_file, condition):
    sentences = read_lines(src_file, 'annotations' in src_file)
    
    sentences = [s[1] for s in sentences if condition(s[1])]
    print sentences[0]
    f1 = open(output_file, 'w')
    for s in sentences:
        f1.write(s+'\n')
    f1.close()

def compare_files_boolean(file1, file2, condition):
    sentences1 = open(file1).read().split('\n')
    sentences2 = open(file2).read().split('\n')

    sentences1_true = len([s for s in sentences1 if condition(s)])
    sentences2_true = len([s for s in sentences2 if condition(s)])

    print 'for file 1, ', sentences1_true/len(sentences1), '% satisfies condition' 
    print 'for file 2, ', sentences2_true/len(sentences2), '% satisfies condition'

def has_person_and_place(sentence):
    print sentence
    try:
        sentence = nlp(sentence)
    except:
        pass
    flag_person = False
    flag_GPE = False
    for s in sentence:
        if s.ent_type_ == u'PERSON':
            flag_person = True
        if s.ent_type_ == u'GPE':
            flag_GPE = True
    return flag_person and flag_GPE

if __name__ == '__main__':
    nlp = spc.load('en')
    print "hi"

#    compare_files_boolean('corrects','false', )
    
    train_file, annotation_file, dev_file, dev_annotation = sys.argv[1:]

    create_file(annotation_file, 'False_person_n_GPE', lambda(x): has_person_and_place(x) and not 'Live_In' in x)
    create_file(annotation_file, 'True_no_person_n_GPE', lambda(x): (not has_person_and_place(x)) and 'Live_In' in x)

    num_correct = len(open('corrects').read().split('\n'))
    num_false = len(open('false').read().split('\n'))
    print num_correct, num_false, num_correct+num_false

    #initialize train_set
    train_set = []
    for sent_id, sent_str in rb.read_lines(train_file):
        sent = nlp(sent_str)
        train_set.append(sent) #sent_str.encode("ascii")) #TODO correct this stuff



    #choose the way to represent the sentences - by default, just the words

    train_set = add_tags(train_set, annotation_file, 'Live_In')

    number_of_true = len([s for s in train_set if s[1]])
    print number_of_true
    print len([s for s in train_set if (not has_person_and_place(s[0])) and 'Live_In' in s[0].text]), 'false negatives'
    print len([s for s in train_set if has_person_and_place([0]) and not 'Live_In' in s[0].text]), 'false positives'
    exit(0)
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
