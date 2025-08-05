# Overview
This repository presents the code for DSP, a novel measure designed to quantify structural polarization in social and information networks accurately. 
DSP improves upon existing metrics by:
- Correcting for biases: It distinguishes genuine polarization from random network features, unlike previous methods that often yield misleadingly high scores on random graphs.
- Integrating a null model: DSP is engineered to have an expected value of zero on random graphs, providing a clearer signal of true polarization.

Validated on both synthetic benchmarks and real-world datasets, DSP provides a more reliable and interpretable tool for understanding societal fragmentation.

# Content
    data/            ... datasets used in the experimental evaluation
    src/             ... Python scripts
    environment.yml  ... required Python libraries

## Source Files
The *src* folder includes the following files:

 - *dk_models.py*: dk-series null models as implemented in [1].
 - *config.py*: configuration file to set the value of the (hyper)parameters.
 - *generators.py*: algorithms to sample from the *G(n,p,l)* null model.
 - *partition_algorithms.py*: partitioning algorithms to find the two communities in the input graph.
 - *polarization_algorithms.py*: implementation of several structural polarization measures.
 - *polaris*: Polaris null model (https://github.com/lady-bluecopper/Polaris) [2].
 - *utils.py*: some useful methods.
 - *run_exp_all_measures_gnml.py*: compute the structural polarization measures on random graphs sampled from the *G(n,p,l)* null model.
 - *run_exp_rwc_congress.py*: compute DSP on the temporal graphs created from US bill co-sponsorship data and roll-call voting data.
 - *run_exp_all_measures_polaris.py*: compute the structural polarization measures on random graphs generated using Polaris.
 - *run_exp_all_measures_real_graphs.py*: compute the structural polarization measures on real networks.
 - *run_exp_all_measures_sbm.py*: compute the structural polarization measures on random graphs generated using the Stochastic Block Model.
 - *run_dk_analysis.py*: compute the normalized structural polarization measures on real networks as proposed in [1].
 - *run_exp_approx_rwc_real_graphs.py*: compute the approximate variant of DSP on real networks.
 - *run_exp_simple_graphs.py*: compute structural polarization measures on the benchmark graph topologies described in the paper.
 - *run_dk_analysis_simple_graphs.py*: compute the normalized structural polarization measures on the benchmark graph topologies described in the paper.
 - *run_exp_given_communities.py*: compute the structural polarization measures on real networks when the node partitions are available.

 ## Parameters

The configuration file allows the specification of the following parameters:
- *n*: number of nodes for the *G(n,p,l)* null model.
- *num_samples*: number of random graphs to generate.
- *average_degree_list*: list of average degrees to consider to set *p* in *G(n,p,l)*: *p = avg_deg / (n - 1)*.
- *skews*: list of relative sizes of the largest partition to consider in *G(n,p,l)*: the size of the largest partition of (randomly selected) nodes is *skew* x *n*.
- *directed*: whether the graphs should be considered as directed or undirected.
- *wbs*: list of values to use to set *wb* in the SBM.
- *bbs*: list of values to use to set *bb* in the SBM.
- *max_blocks*: maximum number of blocks to create in the SBM.
- *sizes*: list of numbers of nodes to consider in the benchmark graph topologies.
- *plen*: length of the path connecting the two clusters in the barbell graph topologies.
- *max_workers*: number of workers for parallel computation.
- *k*: number of influencers for RWC (it will be divided by 100 for ARWC/DP)
- *alpha*: follow-through probability for Personalized PageRank
- *verbose*: printing level
- *pa_method*: method used to partition the graph (rsc, klin, metis).
- *scores*: DSP variants to consider ('v9' and 'v9_mc', where the latter is to be used when the graph has more than 2 labels).
- *all_pol_measures*: whether we want to run the POLARIS experiment for all the polarization measures or only DSP
- *sample_sizes*: list of fractions of nodes to sample in the approximate DSP experiment.
- *data_dir*: directory with the input graph files.
- *out_dir*: directory where the output files will be stored.
- *graph_names*: list of input graph files to consider. If empty, the measures are computed on any file in the directory with extension *extension*.
- *exp_name*: experiment name. It will be used to name the output files for the experiments on real graphs.
- *header*: whether the input file has a header.
- *sep*: separator in the input files.
- *congr_gtype*: which type of US Congress dataset to consider in the congress experiment (roll_call_votes_restr, congress).
- *congr_nums*: list of congress numbers to consider in the congress experiment.
- *congr_types*: list of chambers to consider in the congress experiment (among *s* and *h*).
- *generate*: whether we need to generate random graphs using Polaris.
- *compute_pr*: whether we need to run Personalized PageRank.
- *save_pr*: whether we need to store the Personalized PageRank values (so that they can be loaded from disk in the following experiments).
- null_models = ['1k']: list of null models to consider in Salloum's dk-series experiment.

# Input Format
The file with the network edges should contain one edge per row.
- Unweighted Edges: source_node {separator} destination_node
- Weighted Edges: source_node {separator} destination_node {separator} weight

The separator symbol and information about whether a header row is present must both be specified in your configuration file *config.py*.

Node IDs will be remapped in the range [0, ..., n - 1].

If you have node partition information, provide it in two separate files with the following naming convention:

    community1_<graph_name>.txt
    community2_<graph_name>.txt

Here, *graph_name* should be the exact name of your input graph file (without its extension). Each line in these files should contain the node ID of a node belonging to that specific partition or community.

The folder data includes some of the datasets used in our experimental evaluation of Polaris.

# Requirements
To run the Python scripts in the folder src, you must install the libraries listed in the file environment.yml.

A conda environment can be easily created by running the following commands:

```sh
conda env create -f environment.yml
conda activate pol
```
To use the METIS partitioning algorithm, you must install the METIS library available at this link: https://github.com/KarypisLab/METIS.

# License
This package is released under the GNU General Public License.

# References

[1] Ali Salloum, Ted Hsuan Yun Chen, and Mikko Kivelä. 2022. Separating Polarization from Noise: Comparison and Normalization of Structural Polarization Measures. ACM HCI.

[2] Giulia Preti, Matteo Riondato, Aristides Gionis, and Gianmarco De Francisci Morales. 2025. Polaris: Sampling from the Multigraph Configuration Model with Prescribed Color Assortativity. ACM WSDM.
