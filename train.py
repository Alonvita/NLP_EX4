import sys
import spacy as spc
from collections import Counter, defaultdict
import random
import eval as ev
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

def train(trainX,trainY,devX,devY):
    if not trainX or not trainY or not devX or not devY:
        trainX = devX =  [[[1,2,3,4],[5,6,7],[5,6,8]]]
        trainY = devY = [[1.,0.]]

    train_data = zip(trainX,trainY)
    dev_data = zip(dev_X,dev_Y)
        
    for e in xrange(EPOCHS):
        random.shuffle(train_data)
        train_output=[]
        print train_data
        for inp_vector, tag in train_data:
            print inp_vector
            output = model(inp_vector)

            train_output.append(np.argmax(output.npvalue()))

            true_result = dy.vecInput(2)
            true_result.set(tag)
            loss = dy.binary_log_loss(output,true_result) #-dy.log(dy.pick(output, tag))
            loss.backward()
            trainer.update()
        ev.accuracy(train_output,[y.index(1.) for y in trainY])

        dev_output = []
        for inp_vector, tag in dev_data:
            print inp_vector
            output = model(inp_vector)

            dev_output.append(np.argmax(output.npvalue()))
        ev.accuracy(dev_output,[y.index(1.) for y in devY])

if __name__ == '__main__':
    model = RE_network(20,20,20,1,0.3,12)
    trainer = dy.AdamTrainer(model.model)

    #TODO add train_data of a sort
    #train_data = sheker

    train(None,None,None,None)

    if len(sys.argv) != 3:
        print "proper usage: python extract.py file1 file2"
        exit(0)
