UNKNOWN = '_UNK_'
C_PATH = 1
D_PATH = 2

""" Begin: DIRECTIONS DICT """

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
ROOT = "ROOT"

DIRECTIONS = {
    UP: 'U',
    DOWN: 'D',
    LEFT: 'L',
    RIGHT: 'R',
    ROOT: 'S'
}

""" End: DIRECTIONS DICT """

""" Begins: EXTRACT PATH FUNCTIONS DEFAULTS AND LAMBDA FUNCTIONS """
HEAD_INDEX = 0

PATH_INDEX_OFFSET = 3
NER_INDEX_OFFSET = 0

WORD_OFFSET = 1
POS_OFFSET = 2
DEP_OFFSET = 4

NER_INDEX = lambda (ner): ner[-1]
""" End: EXTRACT PATH FUNCTIONS DEFAULTS AND LAMBDA FUNCTIONS """

""" RELATIONS """

WORK_FOR = 'Work_For'
LIVE_IN = 'Live_In'

anno2i = {UNKNOWN: 0, WORK_FOR: 1, LIVE_IN: 2}

MODEL_NAME = 'cactus_model'
DICTS_FP = 'cactus_dicts'


def extract_ners_from_sentence(sentence):
    """
    extract_ners_from_sentence(sentence).
        Extracts all the Named Entitie Recognitions from a BIO based sentence.
    """
    extracted_ners = []
    last_ner = []

    for index, word in enumerate(sentence):
        if word[5] == 'B':  # BEGIN NER
            if 0 < len(last_ner):
                extracted_ners.append(last_ner)
            last_ner = [index]

        elif word[5] == 'I':  # INSIDE NER
            assert len(last_ner) != 0  # Illegal NER annotation: INSIDE without BEGIN
            last_ner.append(index)
        else:  # OUTSIDE (O)
            if len(last_ner) > 0:
                extracted_ners.append(last_ner)
                last_ner = []

    return extracted_ners


def climb_path_and_extract_ids_to_outpath(sentence, ner_from_sentence, outpath):
    """
    climb_path_and_extract_ids_to_outpath(sentence, ner_from_sentence, outpath).
        Self explanatory.
    """
    """ Begin: FUNCTION DEFAULTS """
    HEAD_INDEX = 0

    PATH_INDEX_OFFSET = 3
    NER_INDEX_OFFSET = 0
    """ End: FUNCTION DEFAULTS """

    # climb up the path, until reacing the head
    while ner_from_sentence[PATH_INDEX_OFFSET] != HEAD_INDEX:  # not HEAD
        outpath.append(ner_from_sentence[NER_INDEX_OFFSET])  # append the ID to the path
        ner_from_sentence = sentence[ner_from_sentence[PATH_INDEX_OFFSET] - 1]  # climb up the path


def lowest_common_ancestor(path1, path2):
    """
    lowest_common_ancestor(path1, path2).
    """
    min_ancestor = None
    ids_down_set = set(path1)

    for id in path2:
        if id in ids_down_set:
            min_ancestor = id
            break

    return min_ancestor


def extract_constituent_path(sentence, ner1, ner2):
    loc1 = ner1[-1]
    loc2 = ner2[-1]

    up_path_indexes_list = []
    down_path_indexes_list = []

    # create paths
    climb_path_and_extract_ids_to_outpath(sentence, sentence[NER_INDEX(ner2)], up_path_indexes_list)
    climb_path_and_extract_ids_to_outpath(sentence, sentence[NER_INDEX(ner1)], down_path_indexes_list)

    # mark end
    up_path_indexes_list.append(0)
    down_path_indexes_list.append(0)

    # Find lowest common ancestor
    ancestor = lowest_common_ancestor(down_path_indexes_list, up_path_indexes_list)

    remove_after_index = up_path_indexes_list.index(ancestor)
    up_path_indexes_list = up_path_indexes_list[:remove_after_index + 1]

    remove_after_index = down_path_indexes_list.index(ancestor)
    down_path_indexes_list = down_path_indexes_list[:remove_after_index + 1][::-1]

    path_up_str = ' '.join('%s-%s' % ((sentence[i - 1][POS_OFFSET] if i != 0 else ROOT), DIRECTIONS[UP]) for i in
                           range(1, len(up_path_indexes_list)))  # POS UP
    path_down_str = ' '.join('%s-%s' % ((sentence[i - 1][POS_OFFSET] if i != 0 else ROOT), DIRECTIONS[DOWN]) for i in
                             range(1, len(down_path_indexes_list)))

    return path_up_str, path_down_str


def extract_typed_dependency_path(sentence, ner1, ner2):
    left_path_indexes_list = []
    right_path_indexes_list = []

    # create paths
    climb_path_and_extract_ids_to_outpath(sentence, sentence[NER_INDEX(ner2)], left_path_indexes_list)
    climb_path_and_extract_ids_to_outpath(sentence, sentence[NER_INDEX(ner1)], right_path_indexes_list)

    # Find lowest common ancestor
    ancestor = lowest_common_ancestor(left_path_indexes_list, right_path_indexes_list)

    if ancestor is None:
        return None

    remove_after_index = right_path_indexes_list.index(ancestor)
    right_path_indexes_list = right_path_indexes_list[:remove_after_index][::-1]

    remove_after_index = left_path_indexes_list.index(ancestor)
    left_path_indexes_list = left_path_indexes_list[:remove_after_index][::-1]

    # create word -> dep dicts
    path_left = ' '.join('%s-%s-%s' % (sentence[i][WORD_OFFSET], DIRECTIONS[LEFT], sentence[i][DEP_OFFSET]) for i in
                         range(0, len(left_path_indexes_list)))
    path_right = ' '.join('%s-%s-%s' % (sentence[i][WORD_OFFSET], DIRECTIONS[RIGHT], sentence[i][DEP_OFFSET]) for i in
                          range(0, len(right_path_indexes_list)))

    common = sentence[ancestor - 1][2]
    return path_left, path_right


def process_sentence(sentence):
    """
    process_sentence(sentence).

    """

    result = {}
    ners = extract_ners_from_sentence(sentence)

    """ Begin: FUNCTION IMPORTS AND LAMBDA OPERATIONS """
    import itertools
    EXTRACT_NER_NAME = lambda (sentence_ner_tup): " ".join([sentence_ner_tup[0][n][1] for n in sentence_ner_tup[1]])
    EXTRACT_NER_TYPE = lambda (sentence_ner_tup): sentence_ner_tup[0][sentence_ner_tup[1][0]][-1]
    EXTRACT_NER_TAG = lambda (sentence_ner_tup): sentence_ner_tup[0][sentence_ner_tup[1][0]][2]
    """ End: FUNCTION IMPORTS AND LAMBDA OPERATIONS """

    ner_pairs = list(itertools.combinations(ners, 2))

    for tmpNer1, tmpNer2 in ner_pairs:
        for ner1, ner2 in [(tmpNer1, tmpNer2), (tmpNer2, tmpNer1)]:  # pair, reverse_pair
            ner1_name = EXTRACT_NER_NAME((sentence, ner1))
            ner2_name = EXTRACT_NER_NAME((sentence, ner2))
            ner1_entity_type = EXTRACT_NER_TYPE((sentence, ner1))
            ner2_entity_type = EXTRACT_NER_TYPE((sentence, ner2))
            ner1_tag = EXTRACT_NER_TAG((sentence, ner1))
            ner2_tag = EXTRACT_NER_TAG((sentence, ner2))

            dep_path = extract_typed_dependency_path(sentence, ner1, ner2)
            constituent_path = extract_constituent_path(sentence, ner1, ner2)

            result[(ner1_name, ner2_name)] = (
            ner1_entity_type, ner2_entity_type, ner1_tag, ner2_tag, constituent_path, dep_path)

    return result


def read_processed_file(fp):
    """ Begin: INNER FUNCTION DEFAULTS """
    OFFSETS = {
        "ID": 0,
        "WORD": 1,
        "POS": 3,
        "HEAD": 5,
        "TREE": 6,
        "NER_P": 7,
        "NER": 8
    }

    ID = "ID"
    WORD = "WORD"
    POS = "POS"
    HEAD = "HEAD"
    TREE = "TREE"
    NER_P = "NER_P"
    NER = "NER"

    """ End: INNER FUNCTION DEFAULTS """

    MAX_ARRAY_LEN = 8

    features_to_id_dict = {}

    with open(fp) as file:
        # Local Variables
        sentence = []
        last_line_was_empty = True
        id = None

        # for each line in the file
        for line in file:
            # strip line
            line = line.strip()

            if line.startswith("#"):
                # take the id, should the line contain it
                if line.__contains__("#id"):
                    id = line.split()[-1]
                continue  # Comment, skip

            # end of sentence
            if len(line) == 0:
                # empty line was not seen
                # process features and restart sentence
                if not last_line_was_empty:
                    features = process_sentence(sentence)
                    features_to_id_dict[id] = features
                    sentence = []
                # set last_line_was_empty to true
                last_line_was_empty = True
                continue

            # NOT an end of sentence indicator
            if 0 < len(line):
                last_line_was_empty = False
                arr = line.split()
                ner_type = arr[OFFSETS[NER]] if MAX_ARRAY_LEN < len(arr) else None

                # initialize the word
                word = (
                    int(arr[OFFSETS[ID]]),
                    arr[OFFSETS[WORD]],
                    arr[OFFSETS[POS]],
                    int(arr[OFFSETS[HEAD]]),
                    arr[OFFSETS[TREE]],
                    arr[OFFSETS[NER_P]],
                    ner_type)

                # append the word to the sentence
                sentence.append(word)
    # return value
    return features_to_id_dict


def read_annotations_file(fp):
    ID_OFFSET = 0
    NER1_OFFSET = 1
    RELATIONSHIP_OFFSET = 2
    NER2_OFFSET = 3

    LAST_SENTENCE_MEMBER_OFFSET = 5

    relation_by_sent_id = {}

    with open(fp) as anootations_file:
        for line in anootations_file:
            line = line.strip()
            arr = line.split('\t', LAST_SENTENCE_MEMBER_OFFSET)  # Last element in the whole sentence

            id = arr[ID_OFFSET]
            ner1 = arr[NER1_OFFSET]
            relation = arr[RELATIONSHIP_OFFSET]
            ner2 = arr[NER2_OFFSET]

            # create a dict for every id that is not in the relation_by_sent_id dict
            if id not in relation_by_sent_id:
                relation_by_sent_id[id] = {}

            # add the ners as a tuple to the id
            relation_by_sent_id[id][(ner1, ner2)] = relation
    return relation_by_sent_id


def compute_feature_key_to_anno_key(anno_by_sent_id, features_by_sent_id):
    # Number of sentences should be the same
    # assert len(features_by_sent_id) == len(anno_by_sent_id)
    sent_ids = set.union(set(features_by_sent_id.keys()), set(anno_by_sent_id.keys()))

    # For each annotation, find it's features from the input
    # Note: They are not always the same :-(
    # i.e "United States" in .annotations is "the United States" in .processed
    from difflib import SequenceMatcher

    feature_key_to_anno_key = {}
    SIM_THRESHOLD = 0.7
    removed_anno_count = 0
    added_anno_count = 0

    for sent_id in sent_ids:
        for anno_key in anno_by_sent_id.get(sent_id, {}):
            anno_ner1, anno_ner2 = anno_key
            found_f_key = None
            f_key_score = 0.0
            both_passed_threshold = False
            both_passed_shared_word = False

            for f_key in features_by_sent_id.get(sent_id, {}):
                f_ner1, f_ner2 = f_key
                ner1_sim = SequenceMatcher(None, anno_ner1, f_ner1).ratio()
                ner2_sim = SequenceMatcher(None, anno_ner2, f_ner2).ratio()
                if f_key_score < ner1_sim + ner2_sim:
                    f_key_score = ner1_sim + ner2_sim
                    found_f_key = f_key

                    both_passed_threshold = SIM_THRESHOLD < SIM_THRESHOLD and ner2_sim < ner1_sim
                    both_passed_shared_word = \
                        len(set(anno_ner1.replace(".", "").split()) & set(f_ner1.replace(".", "").split())) > 0 and \
                        len(set(anno_ner2.replace(".", "").split()) & set(f_ner2.replace(".", "").split())) > 0

            """ 
                Uncomment to see warnings of low-percent matching. I chose to remove those without at least one shared word
                I observed that if a NER exists in the possibilities it would choose it correctly, if it doesn't it chooses a bad one
                But both_passed_shared_word would be False on those occasions

            if not both_passed_threshold:
                print("WARNING: match for annotation key didn't pass threshold")
                print("Sentence id: "+sent_id)
                print("Selected match: "+str(anno_key)+" -> "+str(found_f_key))
                print("Possible matches: "+str(set([a for a,b in features_by_sent_id[sent_id].keys()])))
                if not both_passed_shared_word:
                    print("WARNING: extra low rating. Consider filtering out")
                print("\n")
            """
            if sent_id not in feature_key_to_anno_key:
                feature_key_to_anno_key[sent_id] = {}

            if both_passed_shared_word:
                if found_f_key in feature_key_to_anno_key[sent_id]:
                    print("Warning! double annotation for sentence: " + sent_id + " skipping.\n")
                else:
                    feature_key_to_anno_key[sent_id][found_f_key] = anno_key
                    added_anno_count += 1
            else:
                print("Sentence id: " + sent_id)
                print("Removed match: " + str(anno_key) + " -> " + str(found_f_key))
                print("")
                removed_anno_count += 1

    assert added_anno_count == sum([len(feature_key_to_anno_key[k]) for k in feature_key_to_anno_key])
    print("Found: {} annotations. Removed (because could not find match): {}"
          .format(added_anno_count, removed_anno_count))
    return feature_key_to_anno_key


def convert_features_to_numbers(features_by_sent_id, anno_by_sent_id, feature_key_to_anno_key, Counters=None):
    from utils import StringCounter
    import numpy as np
    sent_ids = features_by_sent_id.keys()
    features_dim_count = len(features_by_sent_id[sent_ids[0]].values()[0])

    allowed_anno = {"Work_For", "Live_In"}

    if Counters is None:
        Counters = np.ndarray(shape=(features_dim_count + 1), dtype=object)
        for i in range(features_dim_count + 1):
            Counters[i] = StringCounter()
        for sent_id in sent_ids:
            for f_key, features in features_by_sent_id[sent_id].items():
                [Counters[i].get_id_and_update(f) for i, f in enumerate(features)]
        # Filter rare features
        for i in range(features_dim_count):
            Counters[i].filter_rare_words(2)  # Min appearance: 2
            Counters[i] = StringCounter(Counters[i].S2I.keys(), "*UNKNOWN*")

    X = []
    Y = []
    YCounter = Counters[-1]
    removed_anno_count = 0
    for sent_id in sent_ids:
        for f_key, features in features_by_sent_id[sent_id].items():
            features_as_ids = tuple([Counters[i].get_id(f) for i, f in enumerate(features)])
            X.append(features_as_ids)
            anno_key = feature_key_to_anno_key[sent_id].get(f_key, None)
            anno = anno_by_sent_id[sent_id].get(anno_key, None)
            if anno == None:
                anno = "None"
            else:
                if anno not in allowed_anno:
                    anno = "None"
                    removed_anno_count += 1
            Y.append(YCounter.get_id_and_update(anno))

    print("Removed {} annotations, because they are not in {}".format(removed_anno_count, allowed_anno))

    return Counters, X, Y


def feat2vec(features, dicts):
    """ Begin: LAMBDA EXPRESSIONS """
    STRIPSPLIT = lambda (str): str.strip().split()
    """ End: LAMBDA EXPRESSIONS """

    word_to_index, tag_to_index, ner_to_index, dep_to_index, annotation_to_index = dicts

    vec = []
    entity_type_1, ner_tag_1, entity_type_2, ner_tag_2, constituent_path, dep_path = features

    named_entities = [ner_to_index.get(entity_type_1, 0),
                      tag_to_index.get(ner_tag_1, 0),
                      ner_to_index.get(entity_type_2, 0),
                      tag_to_index.get(ner_tag_2, 0)]
    vec.append(named_entities)

    cpath = []
    dpath = []

    def create_path(outpath, path, type):
        if path is not None:
            path1 = STRIPSPLIT(path[0])
            path2 = STRIPSPLIT(path[1])

            for path in [path1, path2]:
                for step in path:
                    if type == C_PATH:
                        cons_tag, cons_dir = step.rsplit('-', 1)
                        outpath.extend([tag_to_index.get(cons_tag, 0), annotation_to_index[cons_dir]])
                    else:
                        dep_word, dep_dir, dep = step.rsplit('-', 2)
                        outpath.extend(
                            [word_to_index.get(dep_word, 0), annotation_to_index[dep_dir], dep_to_index.get(dep, 0)])

    create_path(cpath, constituent_path, C_PATH)
    create_path(dpath, dep_path, D_PATH)

    vec.append(cpath)
    vec.append(dpath)
    return vec


def create_predictions_dict(features_by_sent_id, network, i2anno, dicts):
    predictions_dict = {}

    for sent_id in features_by_sent_id:
        for f_key, features in features_by_sent_id[sent_id].items():
            # if we want to predict live_in, then its PERSON -> LOC, GPE
            # if we want to predict works_for, then its PERSON -> ORG, PERSON
            # if features[0] != 'PERSON':
            #    continue
            # if features[2] not in ['ORG','LOC', 'GPE', 'PERSON', 'NORP']:
            #    continue
            feat_vec = feat2vec(features, dicts)
            pred = network.create_network_return_best(feat_vec)
            if pred == anno2i[UNKNOWN]:
                continue
            if sent_id not in predictions_dict:
                predictions_dict[sent_id] = set()
            predictions_dict[sent_id].add((f_key, i2anno[pred]))

    return predictions_dict


def main(src):
    import os.path
    if not os.path.isfile(MODEL_NAME):
        print 'Error: Could not file model file.'
        print 'In order to run extract.py, you have to run train.py before.'
        exit()
    if not os.path.isfile(DICTS_FP):
        print 'Error: Could not file dicts file.'
        print 'In order to run extract.py, you have to run train.py before.'
        exit()

    import pickle
    with open(DICTS_FP, 'rb') as dicts_file:
        dicts = pickle.load(dicts_file)
    word_to_index, tag_to_index, ner_to_index, dep_to_index, annotation_to_index = dicts

    from dynet_network import Model
    network = Model(len(word_to_index), len(annotation_to_index), len(dep_to_index))
    network.model.populate(MODEL_NAME)

    i2anno = {v: k for k, v in anno2i.iteritems()}
    features_by_sent_id = read_processed_file(src)

    predictions_dict = create_predictions_dict(features_by_sent_id, network, i2anno, dicts)

    ids = sorted([int(s[4:]) for s in predictions_dict])
    ids = ['sent%d' % i for i in ids]

    with open('extract_result', 'w') as out_file:
        for sent_id in ids:
            for p in predictions_dict[sent_id]:
                ners = p[0]
                rel = p[1]
                out_file.write('%s\t%s\t%s\t%s\n' % (sent_id, ners[0], rel, ners[1]))


if __name__ == '__main__':
    main('Corpus.DEV.processed')
