import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

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
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='hw4')
    parser.add_argument('train_input', type = str,                  
                        help = 'path to the training input .tsv file')
    parser.add_argument('validation_input', type = str,                  
                        help = 'path to the validation input .tsv file')
    parser.add_argument('test_input', type = str,
                        help = 'path to the test input .tsv file')
    parser.add_argument('feature_dictionary_input', type = str,
                        help = 'path to the word2vec feature dictionary .txt file')   
    parser.add_argument('formatted_train_out', type = str,
                        help = 'path to output train .tsv file')
    parser.add_argument('formatted_validation_out', type = str, 
                        help = 'path to output validation .tsv file')
    parser.add_argument('formatted_test_out', type = str, 
                        help = 'path to output test .tsv file')
    
    args = parser.parse_args()
    inputFile = (args.train_input, args.validation_input, args.test_input)
    outputFile = (args.formatted_train_out,
                  args.formatted_validation_out,
                  args.formatted_test_out)
    for i in range(3):
        dataFile = load_tsv_dataset(inputFile[i])
        word2vec = load_feature_dictionary(args.feature_dictionary_input)
        trimFile = []
        for data in dataFile:
            trim = []
            for word in data[1].split():
                if word in word2vec.keys():
                    trim.append(word)
            trimFile.append(trim)
        
        totalFeature = []
        for line in trimFile:
            sumFeature = np.array([0] * 300).astype('float64')
            for word in line:
                sumFeature += word2vec[word]
            totalFeature.append(sumFeature / len(line))
        
        
        with open(outputFile[i], 'w') as f:   
            for i in range(len(dataFile)):
                label = dataFile[i][0]            
                f.write('%.6f'%label)
                for item in totalFeature[i]:
                    f.write('\t%.6f'%item)
                f.write('\n')