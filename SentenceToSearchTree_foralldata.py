NER = "ner"
PROPN = "PROPN"
FEATURE_TEXT_INDEX = 1
FEATURE_POS_INDEX = 3
FEATURE_HEAD_INDEX = 4
FEATURE_ENT_INDEX = 7
FEATURE_SONS_INDEX = 8


class SearchTreeNode:
    def __init__(self, data):
        print data
        self._data = data
        self._head = None
        self._sons = list()
        self._visited = False

    def set_visited(self):
        """
        set_visited(self).

        turn to visited.
        """
        self._visited = True

    def was_visited(self):
        """
        was_visited(self).

        :return: true if the node was visited, or false otherwise.
        """
        return self._visited

    def get_data(self):
        """
        get_data(self).

        :return: the data held by this node.
        """
        return self._data

    def add_son(self, node):
        """
        add_son(self, node).

        :param node: a SearchTreeNode to add as a son.
        """

        if type(node) != type(self):
            print "Only type of SearchTreeNode can be added as a son to a SearchTreeNode"
            return

        self._sons.append(node)

    def set_head(self, head):
        """
        add_head(self, head).

        :param head: a SearchTreeNode to add as a head.
        """
        if type(head) != type(self):
            print "Only type of SearchTreeNode can be added as a head to a SearchTreeNode"
            return

        self._head = head

    def get_sons(self):
        """
        get_sons(self).

        :return: the list of sons for this node.
        """
        return self._sons

    def get_head(self):
        """
        get_head(self).

        :return: the head of this node.
        """
        return self._head

    def __eq__(self, other):
        """
        __eq__(self, other).

        :param other: another SearchTreeNode
        :return:
        """
        if type(other) != type(self):
            return False

        return self._data == other.get_data()


def _create_words_id_dict(sentence_as_features_lines):
    """
    _give_features_id(features_in_lines).

    the id represents the line of the word in the database. please notice that a line is a line of FEATURES,
    so that the word will actually be located in: database[id][1]

    :param sentence_as_features_lines: a fine containing lines of features. Assuming that the first feature is the word's ID.
    :return: { word -> id } dictionary for quick access
    """

    result = dict()

    for line in sentence_as_features_lines:
        """
        A new line represents the end of a sentence in the data base we used.
        And since this tagger does not support a full database tagging, an empty line will serve as
        a stopping condition.
        
        If you want to tag a whole database, recall the function using the result as a line number.
        For example, should the func return a dict of size S, call the database from [(S + 1):] to skip the empty line.
        
        then, you will be able to add the returned dict to the current dictionary using the 
        following code:

            my_dict_size = len(my_dict) + 1  # +1 for the empty line
        
            for item in result.items():
                my_dict[item[0]] = item[1] + my_dict_size
        
        """
        if line[0] == '':
            return result

        if line[1] in result:
            result[line[1]].append(int(line[0]))
        else:
            result[line[1]] = [int(line[0])]

    return result


def _find_path_for(source, goal):
    """
    _find_fath_for(features_line_i, features_line_j).

    :param source: a line of features as a source
    :param goal: a line of features as a goal

    :return: a path in the parse tree between the two lines
    """

    if source == None:
        return None, False

    goal_found = False
    path = list()

    # already visited
    if source.was_visited():
        return None, goal_found

    # set to visited
    source.set_visited()

    if source == goal:
        # reached goal -> return (goal's text, True)
        return None, True

    # take source's sons
    sons = source.get_sons()

    # run search for all sons
    for son in sons:
        ret_path, goal_found = _find_path_for(son, goal)  # run search

        # in case goal was found
        if goal_found:
            # insert "DOWN" - cause it came from a sun
            path.append("DOWN")
            # insert the son's text
            path.append(son.get_data()[1])

            # add the path returned
            if ret_path is not None:
                path.extend(ret_path)

            return path, goal_found

    # take source's head
    head = source.get_head()

    ret_path, goal_found = _find_path_for(head, goal)  # run search

    # in case goal was found
    if goal_found:
        # insert "UP" - cause it came from a head
        path.append("UP")
        # insert the head's text
        path.append(head.get_data()[1])

        # add the path returned
        if ret_path is not None:
            path.extend(ret_path)

        return path, True

    # no path was found - return None
    return path, goal_found


def _create_index_dict_for_dict_of_lists(dict_of_lists):
    """
    _create_index_dict_for_dict_of_lists(dict).

    :param dict_of_lists: a dict pointing to lists
    :return: an index dict containing all of the values of the dict as keys and 0 as values
    """
    index_dict = dict()

    for key in dict_of_lists.keys():
        index_dict[key] = 0

    return index_dict


def _turn_sentence_features_to_a_search_tree(sentence_as_features, words_id_map, index_dict, root):
    """
    _turn_sentence_to_search_tree_nodes(features_db).

    :param sentence_as_features: a sentence as features
            please see _net_to_net_path_for_sentence_as_features explanation for more info.
    :param words_id_map: a word to id map
    :param index_dict: a dict of indexes
    :param root: the root of the tree as SearchTreeNode
    """

    """
    # PSEUDO: 
    # 1). create TreeNode for the given root index
    # 2). create all sons as TreeNodes
    # 3). for each son:
    #       set root as head
    #       recall function for son with the right index
    #
    """

    if root is None:
        return

    # get root's line
    line = root.get_data()

    raw_root_text = line[FEATURE_TEXT_INDEX]

    # increase the counter for this word closing it with modulus of the size of the words_id_map for this token
    index_dict[raw_root_text] = (index_dict[raw_root_text] + 1) % len(words_id_map[raw_root_text])

    # for each son
    for raw_son_text in line[FEATURE_SONS_INDEX]:
        index = index_dict[raw_son_text]
        sons_id = words_id_map[raw_son_text][index]  # get the son's id, using the index_dict

        # create a son from the line in the sentence that it's id matches
        # the raw son's text id in the map
        son = SearchTreeNode(sentence_as_features[sons_id])

        # set the new node as a head to the son
        son.set_head(root)

        # add the function call as a son
        root.add_son(
            _turn_sentence_features_to_a_search_tree(
                sentence_as_features,
                words_id_map,
                index_dict,
                son))

    return root


def _find_root_line_in_sentence(sentence_as_features):
    """
    _find_root_index_in_sentence(sentence_as_features).

    :param sentence_as_features: a sentence as lines of features
    :return: the root's raw text
    """
    for line in sentence_as_features:
        if line[FEATURE_HEAD_INDEX] == "ROOT":
            return line

    # None for failure
    return None


def _list_from_search_tree(tree_root):
    """
    _list_from_search_tree(tree_root).

    :param tree_root: a search tree root
    :return: (after recursion) a list containing of all of the tree's nodes
    """
    tree_as_list = list()

    tree_as_list.append(tree_root)

    for son in tree_root.get_sons():
        tree_as_list.extend(_list_from_search_tree(son))

    return tree_as_list


def _node_is_of_type(node, type):
    """
    _node_is_of_type(node, type).

    :param node: a SearchTreeNode
    :param type: a type to look for
    :return: true if the node is of type "type", or false otherwise
    """
    # check ner
    if type == "ner":
        # if the node is not of type ner -> skip it
        if node.get_data()[FEATURE_ENT_INDEX] == "_":
            return False
    else:
        # reaching here means that first_type must be of type POS
        # so we will need to check if node_i's POS == first_type
        if node.get_data()[FEATURE_POS_INDEX] != type:
            return False

    return True


def _create_path_for(sentence_as_features, words_id_map, first_type, second_type):
    """
    _ner_to_ner_path_for_sentence_as_features(sentence_as_features).
    Assumes that the sentence as features holds the following features for each word:
        1). [token ID]
        2). [token text]
        3). [token lemma]
        4). [token pos]
        5). [token dep]
        6). [token head (father)]
        7). [head's pos]
        8). [ENT or "_" if none]
        9). [list of sons]

    :param sentence_as_features:
    :param words_id_map: a dict of words to id's
    :param first_type: source's type
    :param second_type: goal's type

    :return: a list of paths between every two NERs in the sentence
    """
    paths = list()

    raw_root_line = _find_root_line_in_sentence(sentence_as_features)

    """
    # calling _turn_sentence_features_to_a_search_tree with:
    #   1). sentence_as_features -- that is the sentence in lines (each line is features for a word in the sentence)
    #   2). words_id_map -- key: word, value: list of indexes in the sentence. Example: "as" -> "[3, 21]"
    #   3). _create_index_dict_for_dict_of_lists(words_id_map) -- a dict of indexes to find
    #                                                              the right index in words_map_id
    #   4). SearchTreeNode(raw_root_line) -- the line of the sentence's root as a SearchTreeNode
    #
    # The function will return the root of the tree created. To access the whole tree, you will need to
    #  iterate over root.get_sons() recursively. For your convenience, the function _list_from_search_tree(root)
    #  is available to create a list of all of the tree nodes.  
    """
    root = \
        _turn_sentence_features_to_a_search_tree(
            sentence_as_features,
            words_id_map,
            _create_index_dict_for_dict_of_lists(words_id_map),
            SearchTreeNode(raw_root_line))

    tree_as_list = _list_from_search_tree(root)

    for i in range(0, len(tree_as_list) - 1):
        # take a node
        node_i = tree_as_list[i]

        # if node_i is not of type first_type, continue to the next node
        if not _node_is_of_type(node_i, first_type):
            continue

        # reaching here means that first_type == node_i.get_data()[POS]
        # or that first_type == "ner" AND node_i.get_data() is indeed an entity
        # therefore, we will need to iterate over the rest of the items
        for j in range(i+1, len(tree_as_list) - 1):
            node_j = tree_as_list[j]

            # if node_j is not of type second_type, continue to the next node
            if not _node_is_of_type(node_j, second_type):
                continue

            # reaching here means that both nodes are of the given types.
            #  -> try to create a path and append it to the paths list
            paths.append(_find_path_for(node_i, node_j))

    # return the paths list
    return paths


def _combine_sons_to_a_list(raw_sons_list):
    """
    _combine_sons_to_a_list(raw_sons_list).

    :param raw_sons_list: a raw sons list
    :return: a list of sons as words
    """
    sons_list = list()

    if raw_sons_list[0] == "[]":
        return list()

    for raw_son in raw_sons_list:
        sons_list.append(str.strip(raw_son, '[],'))

    return sons_list


def _raw_db_to_workable_db(db):
    """
    _raw_db_to_workable_db(db)

    :param db: a features database.
    :return: the database as a list of words
    """
    raw_db_to_lines = (str.split(db, '\n'))
    ret_val = list()

    for line in raw_db_to_lines:
        ret_val.append(str.split(line, '\t'))

    for index in range(len(ret_val)):
        # empty line for end of sentence
        if ret_val[index][0] == "":
            continue

        #sons_list = _combine_sons_to_a_list(ret_val[index][FEATURE_SONS_INDEX:])

        # remove all of the raw sons data from the list
        ret_val[index] = ret_val[index][0:8]

        # append the sons_list created
        #ret_val[index].append(sons_list)

    # remove the empty line
    ret_val = ret_val[0:len(ret_val) - 1]

    return ret_val


if __name__ == '__main__':
    import sys
    _, file_name = sys.argv
    print "hello"
    for sentence in open(file_name).read().split('\n\n'):
        features_file = sentence #\
#            "0 Israel israel PROPN compound television NOUN GPE []\n1 television television NOUN nsubj rejected VERB _ [Israel]\n2 rejected reject VERB ROOT rejected VERB _ [television, skit]\n3 a a DET det skit NOUN _ []\n4 skit skit NOUN dobj rejected VERB _ [a, by, attacked]\n5 by by ADP prep skit NOUN _ [Tzafir]\n6 comedian comedian NOUN compound Tzafir PROPN _ []\n7 Tuvia tuvia PROPN compound Tzafir PROPN _ []\n8 Tzafir tzafir PROPN pobj by ADP _ [comedian, Tuvia]\n9 that that ADJ nsubj attacked VERB _ []\n10 attacked attack VERB relcl skit NOUN _ [that, public, apathy, by]\n11 public public ADJ dobj attacked VERB _ []\n12 apathy apathy NOUN advcl attacked VERB _ []\n13 by by ADP prep attacked VERB _ [depicting]\n14 depicting depict VERB pcomp by ADP _ [family]\n15 an an DET det family NOUN _ []\n16 Israeli israeli ADJ amod family NOUN _ []\n17 family family NOUN dobj depicting VERB _ [an, Israeli, watching]\n18 watching watch VERB acl family NOUN _ [TV, raged]\n19 TV tv NOUN dobj watching VERB _ []\n20 while while ADP mark raged VERB _ []\n21 a a DET det fire NOUN _ []\n22 fire fire NOUN nsubj raged VERB _ [a]\n23 raged rag VERB advcl watching VERB _ [while, fire, outside]\n24 outside outside ADV advmod raged VERB _ []\n"
        print "byr"
        work_db = _raw_db_to_workable_db(features_file)
        print work_db
        word_to_id_dict = _create_words_id_dict(work_db)
        print word_to_id_dict
        path = _create_path_for(work_db, word_to_id_dict, NER, PROPN)
        print path
