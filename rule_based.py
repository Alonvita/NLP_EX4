import sys
import spacy as spc
from collections import Counter
import random
import numpy as np
import codecs 
import util


def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
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

def p_n_p(sentence):
    flag_person = False
    flag_GPE = False
    for s in sentence:
        if s == u'PERSON':
            flag_person = True
        if s == u'GPE':
            flag_GPE = True
    return flag_person and flag_GPE

if __name__ == '__main__':
    nlp = spc.load('en')
    
    train_file, annotation_file, dev_file, dev_annotation = sys.argv[1:]

    #initialize train_set
    train_set = []
    for sent_id, sent_str in read_lines(train_file):
        sent = nlp(sent_str)
        train_set.append(sent) #sent_str.encode("ascii")) #TODO correct this stuff

    #choose the way to represent the sentences - by default, just the words
    train_set = representation(train_set, 'NER')
    print train_set[0]
    train_set = add_tags(train_set, annotation_file, 'Live_In')

    good =0.
    bad =0.
    real_true = []
    predictrue = []
    for t in train_set:
        if t[1] == True:
            real_true.append(t[0])
        if p_n_p(t[0]) == True:
            predictrue.append(t[0])
        if p_n_p(t[0]) == t[1]:
            good+=1
        else:
            bad+=1
    intersection_size = len([value for value in real_true if value in predictrue])
    print "precision is ", float(intersection_size) / len(predictrue)
    print "recall is ", float(intersection_size) / len(real_true)
    print "accuracy is ", good / (good+bad)
