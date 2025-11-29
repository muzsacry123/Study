import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str)
    parser.add_argument("--index_to_word", type=str)
    parser.add_argument("--index_to_tag", type=str)
    parser.add_argument("--hmmprior", type=str)
    parser.add_argument("--hmmemit", type=str)
    parser.add_argument("--hmmtrans", type=str)
    
    args = parser.parse_args()
    
    args.train_input = 'en_data/train.txt'
    args.index_to_word = 'en_data/index_to_word.txt'
    args.index_to_tag = 'en_data/index_to_tag.txt'
    args.hmmprior = 'init.txt'
    args.hmmemit = 'emit.txt'
    args.hmmtrans = 'trans.txt'
    
    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_idx, tags_to_idx, hmmprior, hmmemit, hmmtrans = get_inputs()
    for line in train_data:
        for i in range(len(line)):
            line[i][0] = words_to_idx[line[i][0]]
            line[i][1] = tags_to_idx[line[i][1]]

    # Initialize the initial, emission, and transition matrices
    initial = np.zeros(len(tags_to_idx))
    emission = np.zeros((len(tags_to_idx), len(words_to_idx)))
    transition = np.zeros((len(tags_to_idx), len(tags_to_idx)))
    # Increment the matrices
    count_line = 0
    for line in train_data:
        for i in range(len(line)):
            item = line[i]
            if i == 0:
                initial[item[1]] += 1
                prior = -1
            emission[item[1], item[0]] += 1
            if prior != -1:
                transition[prior, item[1]] += 1
            prior = item[1]
        count_line += 1
        if count_line == 10000:
            break
            
    # Add a pseudocount
    initial += 1
    emission += 1
    transition += 1
    initial = initial / np.sum(initial)
    emission = emission / np.sum(emission, 1, keepdims = True)
    transition = transition / np.sum(transition, 1, keepdims = True)
    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter="\t" for the matrices)
    np.savetxt(hmmprior, initial, delimiter = ' ')
    np.savetxt(hmmemit, emission, delimiter = ' ')
    np.savetxt(hmmtrans, transition, delimiter = ' ')
    