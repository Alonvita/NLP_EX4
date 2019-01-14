import codecs 
import spacy 
import sys


nlp = spacy.load('en')

def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-","(")
        sent = sent.replace("-RRB-",")")
        yield sent_id, sent

def main(train_file):
    f1 = open('metouyag','w')
    sentences = []
    for sent_id, sent_str in read_lines(sys.argv[1]):
        sentence_info = ''
        sent = nlp(sent_str)
        #print("#id:",sent_id)
        #print("#text:",sent.text)
        for word in sent:
            head_id = str(word.head.i+1)        # we want ids to be 1 based
            if word == word.head:               # and the ROOT to be 0.
                assert(word.dep_=="ROOT"),word.dep_
                head_id = "0" # root
            f1.write("\t".join([str(word.i+1), word.text, word.lemma_, word.tag_, word.pos_, head_id, word.dep_, word.ent_iob_, word.ent_type_])+'\n')
            #print "\t".join([str(word.i+1), word.text, word.lemma_, word.tag_, word.pos_, head_id, word.dep_, word.ent_iob_, word.ent_type_])+'\n'
        sentences.append([sent_id, sent])
        f1.write('\n')
    f1.close()

    return sentences
    
##    print "#, Noun Chunks:"
##    for np in sent.noun_chunks:
##        print(np.text, np.root.text, np.root.dep_, np.root.head.text)
##    print "#, named entities:"
##    for ne in sent.ents:
##        print(ne.text, ne.root.ent_type_, ne.root.text, ne.root.dep_, ne.root.head.text)


