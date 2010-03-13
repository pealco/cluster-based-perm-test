from numpy import *
import tables
from scipy.stats import ttest_rel
from pprint import pprint
import numpy as np
from multiprocessing import Pool, cpu_count
from progressbar import ProgressBar
from neighbors import NEIGHBORS
import warnings
warnings.filterwarnings("ignore")

import psyco
psyco.profile()

def find_clusters(tests_matrix, neighbors_hash):
    
    tests_matrix = tests_matrix.T
    samples, sensors = tests_matrix.shape
    label_matrix = zeros((samples, sensors))
    label_matrix[tests_matrix] = 1
    stack = []
    cluster_label = 2
    
    for sample in range(samples):
        for sensor in range(sensors):
            if label_matrix[sample, sensor] == 1:
                label_matrix[sample, sensor] = cluster_label
                stack.append((sample, sensor))
                while stack:
                    cur_sample, cur_sensor = stack.pop()
                    if cur_sample+1 < samples:
                        if label_matrix[cur_sample+1, cur_sensor] == 1:
                            label_matrix[cur_sample+1, cur_sensor] = cluster_label
                            stack.append((cur_sample+1, cur_sensor))
                            
                    if cur_sample > 1:
                        if label_matrix[cur_sample-1, cur_sensor] == 1:
                            label_matrix[cur_sample-1, cur_sensor] = cluster_label
                            stack.append((cur_sample-1, cur_sensor))
                            
                    for neighbor in neighbors_hash[cur_sensor]:
                        if label_matrix[cur_sample, neighbor] == 1:
                            label_matrix[cur_sample, neighbor] = cluster_label
                            stack.append((cur_sample, neighbor))
                                
                cluster_label += 1
                
    return(label_matrix.T - 2)

def compute_statistic_initial(condition_a, condition_b):

    tests = ttest_rel(condition_a, condition_b, axis=2)
    tests = array(tests)
    return tests.swapaxes(0,2)
    
def compute_statistic(condition_a, condition_b):

    tests = my_ttest_rel(condition_a, condition_b, axis=2)
    tests = array(tests)
    return tests.T


def random_partition(epochs):
    samples, sensors, trials = epochs.shape
    random_indexes = random.permutation(range(trials)).reshape(2, -1)
    subset_a = epochs[:, :, random_indexes[0]]
    subset_b = epochs[:, :, random_indexes[1]]
    
    return subset_a, subset_b

def cluster_maxsum(clusters, tests):
    labels = unique(clusters)[1:]
    sums = [sum(tests[..., 0][clusters == label]) for label in labels]
    maxsum = amax(sums)
    return maxsum

def mcmc(epochs, gold_standard, reps, cpu=None):
    golden_standard = (golden_standard == 1)
    random.seed()
    
    mcmc_samples = []
    
    last_cpu = (cpu == cpu_count()-1)
    
    if cpu+1:
        print "Starting MCMC process", cpu+1
    else:
        print "Starting MCMC process ..."
        
    if last_cpu:
        pbar = ProgressBar().start()
    
    for rep in xrange(reps):
        subset_a, subset_b = random_partition(epochs)
        tests = compute_statistic(subset_a, subset_b)
        cluster_of_interest = tests * gold_standard
        
        #maxsum = cluster_maxsum(clusters, tests)
        maxsum = sum(cluster_of_interest)
        mcmc_samples.append(maxsum)
        
        if last_cpu:
            pbar.update(rep/float(reps) * 100)
    
    if last_cpu:
        pbar.finish()
        
    return mcmc_samples

def my_ttest_rel(a, b, axis=0):

    n = a.shape[axis]

    difference = a - b
    variance = difference.var(axis, ddof=1)
    difference_mean = difference.mean(axis)

    t = difference_mean / np.sqrt(variance/float(n))
    t = np.where((difference_mean==0)*(variance==0), 1.0, t) # Define t=0/0 = 1, zero mean and var.

    return t

def cluster_test(a, b, reps, alpha=0.05):
    CPU_COUNTS = cpu_count()
    
    # Start a worker for each processor core.
    pool = Pool(processes=CPU_COUNTS)
    
    comparisons = compute_statistic_initial(a, b)
    significant_comparisons = comparisons[..., 1] <= alpha
    clusters = find_clusters(significant_comparisons, NEIGHBORS)
    cluster_labels = unique(clusters)[1:]
    
    observed_sums = array([sum(comparisons[..., 0][clusters == label]) for label in cluster_labels])
    observed_maxsum = amax(observed_sums)
    
    winner = nonzero(observed_sums == observed_maxsum)
        
    gold_standard = zeros_like(clusters)
    gold_standard[clusters == winner] = 1
    
    winning_cluster = nonzero(clusters == winner)
    winning_cluster_sensors = winning_cluster[0]
    winning_cluster_samples = winning_cluster[1]
    
    grouped_conditions = dstack((a, b))
    
    reps_per_thread = reps / CPU_COUNTS
    results = []
    for cpu in range(CPU_COUNTS):
        result = pool.apply_async(mcmc, (grouped_conditions, gold_standard, reps_per_thread, cpu))
        results.append(result)
    
    mcmc_maxsums = [result.get() for result in results]
    
    reps = array(mcmc_maxsums).size
    print "Observed maxsum:", observed_maxsum
    print "Repetitions:", reps
    print "Cluster samples:", sorted(unique(winning_cluster_samples))
    print "Cluster sensors:", sorted(unique(winning_cluster_sensors))
        
    p_value = sum(mcmc_maxsums > observed_maxsum)/float(reps)
    
    return p_value
    
if __name__ == "__main__":

    data = tables.openFile("R0874_500_denoised.h5")
    epochs = data.root.raw_data_epochs[:,:,17:,:]
    
    print cluster_test(epochs[0,...], epochs[1,...], reps=10000, alpha=0.05)