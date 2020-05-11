from constants import PRUNING_PERCENTAGES

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