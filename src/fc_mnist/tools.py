from constants import PRUNING_PERCENTAGES
import numpy as np

def extract_final_accuracies(set_of_accuracies):
    """
        Extracts the final accuracy from a set of accuracies
    """
    finals = list()
    final_list = set_of_accuracies[len(set_of_accuracies)-1]
    final_accuracy = final_list[len(final_list)-1]
    finals.append(final_accuracy)
    for i in range(len(set_of_accuracies)-1):
        list_ = set_of_accuracies[i]
        final_accuracy = list_[len(list_)-1]
        finals.append(final_accuracy)
    return finals


def generate_percentages(base_percents, lower_bound):
    """
        Generate pruning percentages for each iteration, given the pruning percentages provided
        in constants.py as well as base_percents as argument to this function together with a lower bound
    """
    percentages = {}
    percents = base_percents
    idx = 0
    while percents[1] >= lower_bound:
        new_percents = []
        for i in range(len(base_percents)):
            new_percents.append((1-PRUNING_PERCENTAGES[i])*percents[i])
        
        percentages[idx] = new_percents
        percents = new_percents
        
        idx += 1
    
    return percentages, len(percentages)

def process_es_data(hist_iter, hist_iter_reinit, hist_iter_epochs, hist_iter_reinit_epochs):
    trials = len(hist_iter)
    iterations = len(hist_iter[0])
    epochs = len(hist_iter[0][0])

    # Count average final accuracy for winning ticket
    hist_iter_new = np.zeros(iterations)
    for i in range(trials):
        for j in range(iterations):
            if j == 0:
                final_acc = 0
                for k in range(epochs):
                    if hist_iter[i][len(hist_iter[i])-1][k] == 0:
                        final_acc = hist_iter[i][len(hist_iter[i])-1][k-1]
                        break
                hist_iter_new[j] += final_acc
            else:
                final_acc = 0
                for k in range(epochs):
                    if hist_iter[i][j-1][k] == 0:
                        final_acc = hist_iter[i][j-1][k-1]
                        break
                hist_iter_new[j] += final_acc
    
    hist_iter_new /= trials

    # Count average final accuracy for random reinit
    hist_iter_reinit_new = np.zeros(iterations)
    for i in range(trials):
        for j in range(iterations):
            if j == 0:
                final_acc = 0
                for k in range(epochs):
                    if hist_iter_reinit[i][len(hist_iter_reinit[i])-1][k] == 0:
                        final_acc = hist_iter_reinit[i][len(hist_iter_reinit[i])-1][k-1]
                        break
                hist_iter_reinit_new[j] += final_acc
            else:
                final_acc = 0
                for k in range(epochs):
                    if hist_iter_reinit[i][j-1][k] == 0:
                        final_acc = hist_iter_reinit[i][j-1][k-1]
                        break
                hist_iter_reinit_new[j] += final_acc
    hist_iter_reinit_new /= trials

    # Count average of epochs
    hist_iter_epochs_new = np.zeros(iterations)
    hist_iter_reinit_epochs_new = np.zeros(iterations)
    for i in range(trials):
        for j in range(iterations):
            if j == 0:
                hist_iter_epochs_new[j] += hist_iter_epochs[i][len(hist_iter_epochs[i])-1]
                hist_iter_reinit_epochs_new[j] += hist_iter_reinit_epochs[i][len(hist_iter_reinit_epochs[i])-1]
            else:
                hist_iter_epochs_new[j] += hist_iter_epochs[i][j-1]
                hist_iter_reinit_epochs_new[j] += hist_iter_reinit_epochs[i][j-1]
    hist_iter_epochs_new /= trials
    hist_iter_reinit_epochs_new /= trials

    return hist_iter_new, hist_iter_reinit_new, hist_iter_epochs_new, hist_iter_reinit_epochs_new