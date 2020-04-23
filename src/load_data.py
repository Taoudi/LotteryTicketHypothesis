def unpickle(filename):
    with open('../data/cifar-10-batches-py/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        i = 0
        X = np.zeros(1)
        Y = np.zeros(1)
        for key, val in dict.items():
            i += 1
            if i == 3:
                X = val
            elif i == 2:
                Y = val
    return X, Y
