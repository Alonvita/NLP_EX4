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

def print_tree(sentence):
    for s in sentence:
        toprint = ' - '
        if s.dep_ != 'ROOT':
            head = s.head
            while(head.dep_ != 'ROOT'):
                toprint+= ' - '
                head = head.head
        else:
            toprint=''
        toprint+=s.text
        print toprint, s.dep_

def has_person_head(gpe_ne):
    father = gpe_ne.root.head
    while(father != father.head):
        father = father.head
        if father.ent_type_ == 'PERSON':
            return father.head
    return None

#checks whether the place is a compound of the person
def is_father(person,place):
    father = place.root.head
    while father != father.head:
        if father.ent_type_ == 'PERSON' and father.text in person.text:
            return True
        father = father.head
    return False

def strip_entity(entity):
    string = ''
    for w in entity:
        print w.head
        if w == entity.root or w.head == entity.root:
            string+=w.text+' '
    return string[:-1]

def relations(sentence):
    relations = []
    places = []
    persons =[]
    for ne in sentence.ents:
        if ne.root.ent_type_ == 'GPE':
            places.append(ne)
        if ne.root.ent_type_ == 'PERSON':
            persons.append(ne)
    for pe in persons:
        for pl in places:
            #rule 1 - the place is a compound of the person's entity
            if pl.root.dep_ == 'compound' and is_father(pe,pl):
                pe = strip_entity(pe)
                print pe,pl
                relations.append(str(pe)+'\tLive_In\t'+str(pl))
    return relations

def read_corpus(train_file):
    train_set = []
    id_set = []
    for sent_id, sent_str in read_lines(train_file):
        id_set.append(sent_id)
        sent = nlp(sent_str)
        train_set.append(sent)
    return id_set, train_set    

def main(train_file, output_file):
    if open(train_file).read()[0] == '#':
        print "NOOO!!!! I don't want this format!!! take it away!!!"
        exit()

    #initialize train_set
    id_set, train_set = read_corpus(train_file)

    f1 = open(output_file, 'w')

    for i,t in zip(id_set,train_set):
        for r in relations(t):
            f1.write(i+'\t'+r+'\n')
    f1.close()


nlp = spc.load('en')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "proper usage: python extract.py file1 file2"
        exit(0)

    main(sys.argv[1:])    
    #annotation_file, dev_file, dev_annotation = sys.argv[1:]

##    dev_set = []
##    for sent_id, sent_str in read_lines(dev_file):
##        sent = nlp(sent_str)
##        dev_set.append(sent)


##    #choose the way to represent the sentences - by default, just the words
##    #train_set = representation(train_set, 'NER')
##    train_set = add_tags(train_set, annotation_file, 'Live_In')
##
##    good =0.
##    bad =0.
##    real_true = []
##    predictrue = []
##    for t,i in zip(train_set,id_set):
##        is_there, any_relation = p_n_p(t[0])
##        if t[1]:
##            real_true.append(t[0])
##        if is_there:
##            predictrue.append(t[0])
##        if is_there == t[1]:
##            good+=1
##            if is_there:
##                print "HERE COMES A GOOD ONE"
##                for ne in t[0].ents:
##                    print(ne.text, ne.root.ent_type_, ne.root.text, ne.root.dep_, ne.root.head.text)
##                print i, t[0]
##                print '\n\n\n'
##        else:
##            bad+=1
##            if is_there:
##                print "HERE COMES A SENTENCE"
##                for ne in t[0].ents:
##                    print(ne.text, ne.root.ent_type_, ne.root.text, ne.root.dep_, ne.root.head.text)    
##
##                print_tree(t[0])
##                print i, t[0], '\n\n\n'
##
##    intersection_size = len([value for value in real_true if value in predictrue])
##    precision = float(intersection_size) / len(predictrue)
##    recall = float(intersection_size) / len(real_true)
##    F1 = 2*precision*recall / (precision+recall)
##    
##    print len(real_true), "True"
##    print len(predictrue), "our Truth"
##    print intersection_size, "that one time when our truth is truly true"
##    print "precision is ", precision
##    print "recall is ", recall
##    print "F1 is ", F1
##    print "accuracy is ", good / (good+bad)
