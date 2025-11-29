import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--validation_data", type=str)
    parser.add_argument("--index_to_word", type=str)
    parser.add_argument("--index_to_tag", type=str)
    parser.add_argument("--hmminit", type=str)
    parser.add_argument("--hmmemit", type=str)
    parser.add_argument("--hmmtrans", type=str)
    parser.add_argument("--predicted_file", type=str)
    parser.add_argument("--metric_file", type=str)

    args = parser.parse_args()
    
    args.validation_data = 'en_data/validation.txt'
    args.index_to_word = 'en_data/index_to_word.txt'
    args.index_to_tag = 'en_data/index_to_tag.txt'
    args.hmminit = 'init.txt'
    args.hmmemit = 'emit.txt'
    args.hmmtrans = 'trans.txt'
    args.predicted_file = 'pred.txt'
    args.metric_file = 'metric_10000_val.txt'
    
    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(H):
    if H.ndim == 1:
        H = H.reshape(len(H), 1).T
    v = np.array([])
    for line in H:
        m = np.max(line)
        v = np.append(v, m + np.log(np.sum(np.exp(line-m))))
    return v.reshape(len(v), 1)    

def forwardbackward(seq, loginit, logtrans, logemit):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in
    log_alpha = (loginit + logemit[:, words_to_idx[seq[0][0]]]).reshape([len(tags_to_idx), 1])
    # Initialize log_beta and fill it in
    log_beta = np.zeros(len(tags_to_idx)).reshape([len(tags_to_idx), 1])
    # Compute the predicted tags for the sequence
    for i in range(1, L):
        temp_alpha = (logemit[:, words_to_idx[seq[i][0]]].reshape([len(tags_to_idx), 1])
                      + logsumexp(log_alpha[:, i-1] + logtrans.T))
        log_alpha = np.hstack((log_alpha, temp_alpha))
    for i in range(L - 2, -1, -1):
        temp_beta = (logsumexp(logemit[:, words_to_idx[seq[i+1][0]]]
                     + log_beta[:, 0] + logtrans))
        log_beta = np.hstack((temp_beta, log_beta))
    
    pred = []
    for i in range(L):
        tag = np.argmax(log_alpha[:, i] + log_beta[:, i])
        pred.append(tag.squeeze())
    # Compute the log-probability of the sequence
    log_prob = logsumexp(log_alpha[:, -1])
    # Return the predicted tags and the log-probability
    
    return pred, log_prob.squeeze()
    
    
if __name__ == "__main__":
    # Get the input data
    (val_file, words_to_idx, tags_to_idx, init,
     emit, trans, pred_file, metric_file) = get_inputs()
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    acc = 0
    num_items = 0
    preds = []
    log_probs = []
    for line in val_file:
        pred, log_prob = forwardbackward(line, np.log(init), np.log(trans), np.log(emit))
        for i in range(len(line)):
            if tags_to_idx[line[i][1]] == pred[i]:
                acc += 1
        num_items += len(line)
        preds.append(pred)
        log_probs.append(log_prob)
        
    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    ave_log_prob = np.mean(log_probs)
    acc = acc / num_items
    
    idx_to_tags = {v : k for k, v in tags_to_idx.items()}
    
    with open(metric_file, 'w') as f:
        f.write('Average Log-Likelihood: ' + str(ave_log_prob) + '\n')
        f.write('Accuracy: ' + str(acc))
    
    with open(pred_file, 'w') as f:
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                space_num = (len(val_file[i][j][0]) // 6 + 1) * 6
                f.write(val_file[i][j][0].ljust(space_num) +
                        idx_to_tags[preds[i][j]] + '\n')
            f.write('\n')
    