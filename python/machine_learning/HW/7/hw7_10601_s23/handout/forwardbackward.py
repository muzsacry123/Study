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
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

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
    """
    Implementation of the log-sum-exp trick. H can be either a vector or a matrix.
    """
    
    if len(H.shape) == 1:
        H = H.reshape(len(H), 1).T
    v = np.array([])
    for line in H:
        m = np.max(line)
        v = np.append(v, m + np.log(np.sum(np.exp(line - m))))
    
    return v.reshape(len(v), 1)


def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in - feel free to use words_to_indices to index the specific word
    log_alpha = (loginit + logemit[:,words_to_indices[seq[0][0]]]).reshape(M, 1)

    # Initialize log_beta and fill it in - feel free to use words_to_indices to index the specific word
    log_beta = np.zeros(M).reshape([M, 1])

    # Compute the predicted tags for the sequence - tags_to_indices can be used to index to the rwquired tag
    for i in range(1,L):
        temp_alpha = (logemit[:, words_to_indices[seq[i][0]]].reshape(M,1) + logsumexp(log_alpha[:, i-1] + logtrans.T))
        log_alpha = np.hstack((log_alpha, temp_alpha))
    
    for i in range(L-2, -1, -1):
        temp_beta = (logsumexp(logemit[:, words_to_indices[seq[i+1][0]]] + log_beta[:,0] + logtrans))
        log_beta = np.hstack((temp_beta, log_beta))
    
    prediction = []
    for i in range(L):
        tag = np.argmax(log_alpha[:,i] + log_beta[:,i])
        prediction.append(tag.squeeze())

    # Compute the stable log-probability of the sequence
    log_prob = logsumexp(log_alpha[:,-1])

    # Return the predicted tags and the log-probability
    return prediction, log_prob.squeeze()
    

    
    
if __name__ == "__main__":
    # Get the input data
    (validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file) = get_inputs()

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    accuracy = 0
    items = 0
    predictions = []
    log_probs = []
    
    for line in validation_data:
        pred, log_prob = forwardbackward(line, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit), words_to_indices, tags_to_indices)
        
        for i in range(len(line)):
            if tags_to_indices[line[i][1]] == pred[i]:
                accuracy += 1
        items += len(line)
        predictions.append(pred)
        log_probs.append(log_prob)

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    avg_log_likelihood = np.mean(log_probs)
    accuracy /= items
    
    indices_to_tags = {i: t for t, i in tags_to_indices.items()}
    
    with open(predicted_file, 'w') as f:
        for i, val in enumerate(validation_data):
            for j, (text, _) in enumerate(val):
                space_num = (len(text) // 6 + 1) * 6
                f.write(f"{text.ljust(space_num)}{indices_to_tags[predictions[i][j]]}\n")
            f.write('\n')
    
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: " + str(avg_log_likelihood) + "\n")
        f.write("Accuracy: " + str(accuracy) + "\n")
        
    file = open("train_avg_log_likelihood.txt", 'a')
    file.write(str(avg_log_likelihood) + "\n")
    file.close()
    

        
    