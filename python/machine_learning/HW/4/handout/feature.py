import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    
    # Take care of the inputs and outputs
    inputs = (args.train_input, 
              args.validation_input, 
              args.test_input)
    outputs = (args.train_out,
               args.validation_out,
               args.test_out)
    
    # Handle the data
    for index in range(3):      # since we both have 3 inputs and 3 outputs
        dataSet = load_tsv_dataset(inputs[index])      # load the i-th input file
        gloveMapDict = load_feature_dictionary(args.feature_dictionary_in)       # load the word-to-vector dictionary file
        
        # Trim the dataSet, to include only words in the dictionary
        trimSet = []
        for data in dataSet:    # data is a tuple (label, review) collected from dataSet
            Words = []
            for word in data[1].split():   # split data into words
                if word in gloveMapDict.keys():     # check if the word is in the dictionary
                    Words.append(word)
            trimSet.append(Words)
            
        # Dealing with the final feature vector
        # finalFeature = []
        # for line in trimSet:
        #     sumFeature = np.array([0] * 300).astype('float64')
        #     for word in line:
        #         sumFeature += gloveMapDict[word]
        #     finalFeature.append(sumFeature / len(line))
        
        finalFeature = [np.sum([gloveMapDict[word] for word in line], axis=0) / len(line)
                for line in trimSet]
        
        # Finally, work with the output files
        with open(outputs[index], 'w') as f:   
            for i in range(len(dataSet)):
                label = dataSet[i][0]            
                f.write('%.6f'%label)
                for item in finalFeature[i]:
                    f.write('\t%.6f'%item)
                f.write('\n')