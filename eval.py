import sys

epsilon = 0.0000001

def main(gold_file,pred_file):
    print gold_file, pred_file
    gold_data = [r.split('(')[0][:-1] for r in open(gold_file).readlines()]
    pred_data = [r.split('(')[0][:-1] for r in open(pred_file).readlines()]
    
    print len(gold_data)
    print len(pred_data)

    gold_entities = set(gold_data)
    pred_entities = set(pred_data)


    print len(gold_entities), "real true"
    print len(pred_entities), "our truth"
    print len(gold_entities.intersection(pred_entities)), "when truths collide"
    print [p for p in pred_entities if p not in gold_entities.intersection(pred_entities)]
    print [g for g in gold_entities if 'G. Ernest' in g]
    print [g for g in gold_entities if 'Sirhan' in g]
    prec = len(gold_entities.intersection(pred_entities)) / float(len(pred_entities))
    rec  = len(gold_entities.intersection(pred_entities)) / float(len(gold_entities))
    F1 = prec*rec/(prec+rec + epsilon)
    print "Prec:%s Rec:%s F1:%s" % (prec, rec, F1)


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
