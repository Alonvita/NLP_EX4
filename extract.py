import sys
import spacy as spc
from collections import Counter, defaultdict
import random
import numpy as np
import dynet as dy
import codecs 
import util

LENGTH_OF_NER = 4
EPOCHS = 15

class RE_network(object):
    def __init__(self,words_num,arrows_num,dep_num,dropout,embed_dim,lstm_dim):
        self.dropout = dropout
        self.words_num = words_num
        self.arrows_num = arrows_num
        self.dep_num = dep_num
        self.lstm_dim = lstm_dim

        self._model = dy.ParameterCollection()

        #representing of words as vectors
        self.word_embeds = self.model.add_lookup_parameters((words_num,embed_dim))
        self.arrow_embeds = self.model.add_lookup_parameters((arrows_num,embed_dim))
        self.dep_embeds = self.model.add_lookup_parameters((dep_num, embed_dim))

        #LSTM - to represent the constituent and dependency path between entities
        self.constituent_lstm = dy.LSTMBuilder(1,embed_dim,lstm_dim,self.model)
        self.dependency_lstm = dy.LSTMBuilder(1,embed_dim,lstm_dim,self.model)

        #MLP - 2 inner layers of 128 and 64, followed by a layer of 2
        # softmax will be applied to determine whether the input has the relation we seek 
        layers_size = (128,64)
        self.W1 = self.model.add_parameters((layers_size[0],LENGTH_OF_NER+2*lstm_dim))
        self.b1 = self.model.add_parameters(layers_size[0])
        self.W2 = self.model.add_parameters((layers_size[1],layers_size[0]))
        self.b2 = self.model.add_parameters(layers_size[1])
        self.W3 = self.model.add_parameters((2,layers_size[1]))
        self.b3 = self.model.add_parameters(2)

    def __call__(self,inputs, is_train=True):
        ners, constituent_path, dep_path = inputs

        dy.renew_cg()

        #make ner a dynet expression
        ners_vec = dy.vecInput(LENGTH_OF_NER)
        ners_vec.set(ners)

        #get vector from lstm on constituent path 
        if len(constituent_path) > 0:
            constituent_path = [self.word_embeds[x] if i%2==0 else self.arrow_embeds[x] for i,x in enumerate(constituent_path)]
            if is_train:
                constituent_path = [dy.dropout(x,self.dropout) for x in constituent_path]
            lstm_init1 = self.constituent_lstm.initial_state()
            cons_vec = lstm_init1.transduce(constituent_path)[-1]
        else:
            cons_vec = dy.vecInput(self.lstm_dim)

        #get vector from lstm on dependency path 
        if len(dep_path) > 0:
            dep_vec = []
            for i,x in enumerate(dep_path):
                if i%3==0:
                    dep_vec.append(self.word_embeds[x])
                elif i%3==1:
                  dep_vec.append(self.arrow_embeds[x])
                else:
                  dep_vec.append(self.dep_embeds[x])
            if is_train:
                dep_vec = [dy.dropout(x,self.dropout) for x in dep_vec]
            lstm_init2 = self.dependency_lstm.initial_state()
            dep_vec = lstm_init2.transduce(dep_vec)[-1]
        else:
            dep_vec = dy.vecInput(self.lstm_dim)

        final_input = dy.concatenate([ners_vec,cons_vec,dep_vec])

        return dy.softmax(self.W3*dy.tanh(self.W2*dy.tanh(self.W1*final_input+self.b1)+self.b2)+self.b3)

    @property
    def model(self):
        return self._model


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

def modify_entity(entity, sentence):
    string = ''
    for w in entity:
        if w == entity.root or w.head == entity.root:
            string+=w.text+' '
    start_index = entity[0].i
    end_index = entity[-1].i

    wrong = ['','GPE']

    start = ''
    for w in reversed(sentence[:start_index]):
        if w.text in ['of','the'] or (w.text[0].isupper() and not w.ent_type_ in wrong and not w.text in string):
            start = w.text+' '+start
        else:
            break
    i1 = 0
    for i,c in enumerate(start):
        if c.isupper():
            i1 = i
            break
    start = start[i1:]
    string = start+string 
    for w in sentence[end_index:]:
        if w.text in ['of','the'] or (w.text[0].isupper() and not w.ent_type_ in wrong and not w.text in string):
            string += w.text+' '
        else:
            break
    print string
    return string[:-1]

def unify(relations, sentence):
    tmp_relations = [r.split('\t') for r in relations]
    persons = [r[0] for r in tmp_relations]
    places = [r[-1] for r in tmp_relations]

    toremove = []
    for i,p in enumerate(persons):
        for j,p2 in enumerate([pe for pe in persons if pe != p]):
            if sentence.find(p)+len(p) >= sentence.find(p2) and sentence.find(p) < sentence.find(p2) and places[i]==places[j]:
                person_entity = sentence[sentence.find(p):sentence.find(p2)+len(p2)]
                relations.append(person_entity+'\tLive_In\t'+places[i])
                toremove.append(p+'\tLive_In\t'+places[i])
                toremove.append(p2+'\tLive_In\t'+places[j])
    for r in toremove:
        if r in relations:
            relations.remove(r)
    return list(set(relations))

def have_common_verb(person,place):
    pe = person.root.head
    pl = place.root.head
    pe_not_noun = False
    pl_not_noun = False
    while pe.pos_ != 'VERB' and pe.head != pe:
        if pe_not_noun and pe.pos_ in ['NOUN', 'PROPNN']:
            break
        if pe.pos_ not in ['NOUN', 'PROPNN']:
            pe_not_noun == True
        pe = pe.head
    while pl.pos_ != 'VERB' and pl.head != pl:
        if pl_not_noun and pl.pos_ in ['NOUN', 'PROPNN']:
            break
        if pl.pos_ not in ['NOUN', 'PROPNN']:
            pl_not_noun == True
        if pl.text == 'on':
            break
        pl = pl.head
        
    return pe == pl and pe.lemma_ not in ['kill','die','shoot']
    
def is_type(entity, type):
    return processed[entity.text] == type

def relations(sentence):
    relations = []
    places = []
    persons =[]
    for ne in sentence.ents:
        if ne.root.ent_type_ == 'GPE' and is_type(ne,'GPE'):
            places.append(ne)
        if ne.root.ent_type_ == 'PERSON' and is_type(ne,'PERSON'):
            persons.append(ne)
    for pe in persons:
        for pl in places:
            #rule 1 - the place is a compound of the person's entity
            if pl.root.dep_ == 'compound' and is_father(pe,pl):
                p1 = modify_entity(pe,sentence)
                #p2 = modify_entity(pl,sentence)
                relations.append(p1+'\tLive_In\t'+pl.text)
                continue
            #rule 2 - the place and the person share a verb
            if have_common_verb(pe,pl):
                p1 = modify_entity(pe,sentence)                
                #p2 = modify_entity(pl,sentence)
                relations.append(p1+'\tLive_In\t'+pl.text)
                continue
    relations = unify(relations, sentence.text)
    return relations

def read_corpus(train_file):
    train_set = []
    id_set = []
    for sent_id, sent_str in read_lines(train_file):
        id_set.append(sent_id)
        sent = nlp(sent_str)
        train_set.append(sent)
    return id_set, train_set    

def read_processed(processed_file):
    word_to_ner = {}
    word = ''
    lines = open(processed_file).readlines()
    for i,line in enumerate(lines):
        if line.isspace() or line[0] == '#' or line.split('\t')[7] == 'O':
            continue
        line = line.split('\t')
        word_to_ner[i] = line[8][:-1]
    while word_to_ner:
        for key in word_to_ner.keys():
            if key+1 in word_to_ner.keys() and word_to_ner[key] == word_to_ner[key+1]:
                processed[lines[key].split('\t')[1]+' '+lines[key+1].split('\t')[1]] = word_to_ner[key]
            else:
                processed[lines[key].split('\t')[1]] = word_to_ner[key]
            del word_to_ner[key]

    

def main(train_file, output_file):
    train_file = train_file.replace('.txt','')
    if open(train_file+'.txt').read()[0] == '#':
        print "NOOO!!!! I don't want this format!!! take it away!!!"
        exit()

    #initialize train_set
    id_set, train_set = read_corpus(train_file+'.txt')

    read_processed(train_file+'.processed')
    print processed
    
    f1 = open(output_file, 'w')

    for i,t in zip(id_set,train_set):
        for r in relations(t):
            f1.write(i+'\t'+r+'\n')
    f1.close()


#nlp = spc.load('en')
nlp = spc.load('en_core_web_sm')
processed = defaultdict(lambda:None, { })

if __name__ == '__main__':
    model = RE_network(20,20,20,1,0.3,12)
    trainer = dy.AdamTrainer(model.model)
    print model([[1,2,3,4],[5,6,7],[]]).npvalue()

    #TODO add train_data of a sort

    for e in xrange(EPOCHS):
        output = model([[1,2,3,4],[5,6,7],[5,6,8]])
        tag = dy.vecInput(2)
        tag.set([0.,1,])
        print output.npvalue()
        print tag.npvalue()
        loss = dy.binary_log_loss(output,tag) #-dy.log(dy.pick(output, tag))
        loss.backward()
        trainer.update()

    if len(sys.argv) != 3:
        print "proper usage: python extract.py file1 file2"
        exit(0)
