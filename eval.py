import sys

epsilon = 0.0000001

def main(gold_file,pred_file):
    print gold_file, pred_file
    gold_data = [r.split('(')[0][:-1].replace('.','') for r in open(gold_file).readlines()]
    pred_data = [r.split('(')[0][:-1].replace('.','') for r in open(pred_file).readlines()]
    
    print len(gold_data)
    print len(pred_data)

    accuracy(gold_data,pred_data)

def accuracy(gold,pred):
    gold_entities = set(gold)
    pred_entities = set(pred)


    print len(gold_entities), "real true"
    print len(pred_entities), "our truth"
    print len(gold_entities.intersection(pred_entities)), "when truths collide"
    print [p for p in pred_entities if p not in gold_entities.intersection(pred_entities)]
    prec = len(gold_entities.intersection(pred_entities)) / float(len(pred_entities))
    rec  = len(gold_entities.intersection(pred_entities)) / float(len(gold_entities))
    F1 = 2*prec*rec/(prec+rec + epsilon)
    print "Prec:%s Rec:%s F1:%s" % (prec, rec, F1)

    return F1


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
