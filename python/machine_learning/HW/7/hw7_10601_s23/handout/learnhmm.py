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
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

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
    train_data, words_to_index, tags_to_index, hmmprior, hmmemit, hmmtrans = get_inputs()
    for line in train_data:
        for i in range(len(line)):
            line[i][0] = words_to_index[line[i][0]]
            line[i][1] = tags_to_index[line[i][1]]
            

    # Initialize the initial, emission, and transition matrices
    pi = np.zeros(len(tags_to_index))
    A = np.zeros((len(tags_to_index), len(words_to_index)))
    B = np.zeros((len(tags_to_index), len(tags_to_index)))

    # Increment the matrices
    lineCount = 0
    for line in train_data:
        for i in range(len(line)):
            item = line[i]      # (word_i, tag_i)
            if i == 0:
                pi[line[i][1]] += 1
            else:
                B[line[i-1][1],line[i][1]] += 1
            A[line[i][1],line[i][0]] += 1
        
        lineCount += 1
        if lineCount == 1000000:  break

    # Add a pseudocount and normalize
    # pi = (pi + 1) / np.sum(pi)
    # A = (A + 1) / np.sum(A, axis=1, keepdims=True)
    # B = (B + 1) / np.sum(B, axis=1, keepdims=True)
    pi += 1
    pi = pi / np.sum(pi)
    A += 1
    A = A / np.sum(A, axis=1, keepdims=True)
    B += 1
    B = B / np.sum(B, axis=1, keepdims=True)
    

    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    np.savetxt(hmmprior, pi, delimiter=" ")
    np.savetxt(hmmemit, A, delimiter=" ")
    np.savetxt(hmmtrans, B, delimiter=" ")
    
 
