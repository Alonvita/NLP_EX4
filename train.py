import sys
import spacy as spc
from collections import Counter, defaultdict
import random
import eval as ev
from extract import *
import numpy as np
import dynet as dy
import codecs 
import util

LENGTH_OF_NER = 4
EPOCHS = 15

MODEL_NAME = 'SENE_model'

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

def toInputs(id2features,id2anno,dicts):
    X = []
    Y = []

    feat2anno = compute_feature_key_to_anno_key(id2anno,id2features)
    ids = id2features.keys()
    for s_id in ids:
        for pair,features in id2features[s_id].items():
            X.append(feat2vec(features,dicts))

            if pair not in feat2anno[s_id]:
                Y.append(anno2i[UNKNOWN])
                continue

            anno_key = feat2anno[s_id][pair]
            if anno_key not in id2anno[s_id]:
                Y.append(anno2i[UNKNOWN])
                continue

            anno = id2anno[s_id][anno_key]
            if anno not in anno2i:
                Y.append(anno2i[UNKNOWN])
                continue

            # annontion is allowed, and we know its type.
            Y.append(anno2i[anno])

    print sum([y for y in Y if y==1])
    exit(0)
    return X,Y

def accuracy(gold,pred):
##    from collections import Counter
##    table = {}
##    good = bad = 0.0
##    for pred, gold in zip(pred, gold):
##        if gold not in table:
##            table[gold] = Counter()
##        table[gold].update([pred])
##        if gold == pred:
##            good += 1
##        else:
##            bad += 1
##
##    acc = good / (good + bad)
##    recall = {}
##    prec = {}
##    f1 = {}
##    # move onto computing recall and precision
##    for gold in table:
##        tp = float(table[gold][gold])
##        tpfn = sum(table[gold].values())
##        recall[gold] = tp / tpfn
##
##        sm = 0.0
##        for r_gold in table:
##            sm += table[r_gold][gold]
##        if sm > 0:
##            sm = tp / sm
##        prec[gold] = sm
##
##        denom = (recall[gold] + prec[gold])
##        if denom != 0:
##            f1[gold] = (2.0 * recall[gold] * prec[gold]) / denom
##        else:
##            f1[gold] = 0.0
##
##    return acc, recall, prec, f1
    
    epsilon = 0.00000001
    good = total = true = pred_true = correct_pred = epsilon
    for g,p in zip(gold,pred):
        if g==p:
            good+=1
        total+=1

        if g==1:
            true+=1
        if p==1:
            pred_true+=1
        if g==1 and p==1:
            correct_pred += 1

    print "accuracy is", good/total
    print "precision is", correct_pred/pred_true
    print "recall is", correct_pred/true

def train(train_file,train_anno_file,dev_file,dev_anno_file):

    #train_data
    dicts = get_dicts(train_file)
    id2features = read_processed_file(train_file)
    id2anno = read_annotations_file(train_anno_file)
    trainX,trainY = toInputs(id2features,id2anno,dicts)
    
    id2features = read_processed_file(dev_file)
    id2anno = read_annotations_file(dev_anno_file)
    devX,devY = toInputs(id2features,id2anno,dicts)

    train_data = zip(trainX,trainY)
    dev_data = zip(devX,devY)

    f1 = 0.
    for e in xrange(EPOCHS):
        random.shuffle(train_data)
        train_output=[]
        for inp_vector, tag in train_data:
            output = model(inp_vector)

            train_output.append(np.argmax(output.npvalue()))

##            true_result = dy.vecInput(2)
##            true_result.set(tag)
            loss = -dy.log(dy.pick(output, tag)) #dy.binary_log_loss(output,true_result) #
            loss.backward()
            trainer.update()

        print accuracy(trainY,train_output) #[y.index(1.) for y in trainY],train_output)
        exit(0)
        dev_output = []
        for inp_vector, tag in dev_data:
            output = model(inp_vector)

            dev_output.append(np.argmax(output.npvalue()))
        F1 = accuracy(devY,dev_output)

        if F1 > f1:
            f1 = F1
            model.model.save(MODEL_NAME)

if __name__ == '__main__':
    w2i, t2i, n2i, d2i, a2i = get_dicts(sys.argv[1])
    print len(w2i),len(a2i),len(d2i)
    model = RE_network(len(w2i),len(a2i),len(d2i),0.3,1,12)
    trainer = dy.AdamTrainer(model.model)

    if len(sys.argv) == 5:
        train(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
        print "proper usage: python extract.py file1 file2"
        exit(0)
