import numpy as np

# RANDOM GRAPHS
# num nodes
n = 10000
# num samples
num_samples = 100
# node average degree
average_degree_list = [3, 6, 9]
# percentage of nodes per label R
skews = [0.9, 0.7, 0.5]
# whether the graph should be directed or not
directed = False
# SBM parameters (also uses n and num_samples)
wbs = [0.1, 0.2, 0.3, 0.4, 0.5]
bbs = [0.01, 0.02, 0.03, 0.04, 0.05]
max_blocks = 4 # maximum number of blocks to create
# small controlled networks
sizes = [1000, 2000, 3000, 4000, 5000]
plen = 4

# num of workers for parallel computation
max_workers = 6
# number of influencers for RWC (it will be divided by 100 for ARWC/DP)
k = 10
# alpha for PPR
alpha = 0.85
# printing level
verbose = False
# method used to partition the graph
pa_method = 'metis' # rsc klin metis
# rwc variants
scores = ['v9']
# whether we want to run the POLARIS experiment for all the polarization measures or only RWC++
all_pol_measures = True
# fraction of nodes to sample in the approximate RWC++ experiment
sample_sizes = [.1, .2, .4, .8, 1.]
# whether we need to generate the polaris samples
generate = True
# whether we need to run PPR
compute_pr = True
# whether we need to store the PPR values
save_pr = False
# null models to consider in Salloum dk experiment
null_models = ['1k']

# DATA
data_dir = '../data/'
out_dir = '../out/'

# REAL GRAPHS

## SALLOUM
# graph_names = []
# data_dir = '../data/salloum_data'
# exp_name = 'salloum'
# extension = 'edgelist'
# header = True
# sep = ','
# get_filename = lambda x: f'{x}.edgelist'

## GARIMELLA
# graph_names = []
# graph_names = ['beefban', 'russia_march', 'germanwings', 'onedirection', 'nemtsov', 
#                'netanyahu', 'indiasdaughter', 'baltimore', 'indiana', 'ukraine', 
#                'gunsense', 'leadersdebate', 'sxsw', 'nepal', 'ultralive', 'ff', 
#                'jurassicworld', 'wcw', 'nationalkissingday', 'mothersday']
# data_dir = '../data/garimella_data'
# exp_name = 'garimella'
# extension = 'CC.txt'
# header = False
# sep = ','
# get_filename = lambda x: f'retweet_graph_{x}_threshold_largest_CC.txt'

## TWITTER
# graph_names = []
# data_dir = '../data/twitter_data'
# exp_name = 'twitter'
# extension = 'tsv'
# header = False
# sep = '\t'
# get_filename = lambda x: f'{x}.tsv'

## US CONGRESS
# data_dir = '../data/congress_data'
# congr_gtype = 'roll_call_votes_restr' # roll_call_votes congress roll_call_votes_restr
# exp_name = 'roll_call_votes_restr' # roll_call_votes congress roll_call_votes_restr
# congr_nums = np.arange(93, 115)
# congr_types = ['s', 'h']
# extension = '.tsv'
# header = False
# sep = '\t'

## ICWSM NETWORKS
# graph_names = ['icwsm_mention', 'icwsm_retweet']
# data_dir = '../data/icwsm_data'
# exp_name = 'icwsm'
# extension = 'edgelist'
# header = False
# sep = '\t'
# get_filename = lambda x: f'{x}.edgelist'