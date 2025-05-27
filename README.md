# An Efficient Hybridization of Graph Representation Learning and Metaheuristics for the Constrained Incremental Graph Drawing Problem

Welcome to the GitHub repository for our research paper entitled "An Efficient Hybridization of Graph Representation Learning and Metaheuristics for the Constrained Incremental Graph Drawing Problem." This repository contains the codes of the Greedy Randomized Search Procedure (GRASP) heuristics introduced and implemented by Charytitsch and Nascimento (2024). The authors introduced the Graph Learning GRASP (GL-GRASP) heuristics for the Constrained Incremental Graph Drawing Problem (C-IGDP). To compare with the literature GRASP heuristics, the authors also implemented them in Python and also made them available in this repository.

## Overview

The methods introduced in this paper use information from Graph Representation Learning (GRL) to guide the constructive phase of the GL-GRASP heuristics. The GRASP heuristics are based on the literature heuristics for the C-IGDP, adapted so they employ the graph embedding information in the constructive phase. Computational experiments considered four distinct methods for generating node embeddings: two based on DL, one spectral method, and one based on matrix factorization. Stochastic node embedding strategies had two different learning GRASP versions. The approaches proved promising, particularly those involving DL, that outperformed state-of-the-art methods in denser and with more numerous layers.

## Cite

One have to cite this repository and the following paper for any use of the resources of this repository.

- Bruna Cristina Braga Charytitsch and Mariá Cristina Vasconcelos Nascimento (2024) An Efficient Hybridization of Graph Representation Learning and Metaheuristics for the Constrained Incremental Graph Drawing Problem. Submitted to European Journal of Operational Research.

## Methods and Implementations
The methods utilized in our research and their implementations were derived from the CogDL framework. CogDL provides a comprehensive research toolkit for Deep Learning on Graphs <https://docs.cogdl.ai/en/latest/>. The list of methods:

- **Node2Vec**: From the paper “node2vec: Scalable feature learning for networks” <https://dl.acm.org/doi/10.1145/2939672.2939754>
- **Hope**: From the paper “Grarep: Asymmetric transitivity preserving graph embedding” <https://dl.acm.org/doi/10.1145/2939672.2939751>
- **SDNE**: From the paper  “Structural Deep Network Embedding” <https://dl.acm.org/doi/10.1145/2939672.2939753>
- **SPECTRAL**: From the paper  “Leveraging social media networks for classiﬁcation” <https://link.springer.com/article/10.1007/s10618-010-0210-x>.

## Repository Contents

- **Implementation**: Python implementation of our proposed hybrid approach.
- **Datasets**: Datasets used in our experiments.
- **Results**: Performance metrics from our experiments.
- **Paper**: Link to the full paper discussing our methodology, experimental setup, and findings.

## Getting Started

Feel free to explore our code, datasets, and results and propose improvements. We encourage researchers and professionals interested in Graph Drawing, Machine Learning, and Optimization to delve into our work and contribute to advancing science.

## Operation of the Codes

### Input Parameters

The program requests three input parameters from the user:
1. **Total number of executions**
2. **Total number of GRASP iterations**
3. **Complementary stopping criterion \( η_max \)**: terminates the code execution if \( η_max \) consecutive GRASP iterations occur without improvements.

An example of input parameters can be found in the `input.txt` file. If preferred, these parameters can be read directly from this file instead of being entered manually.

### Folder and File Structure

For the code execution, ensuring that the `res`, `data` folders, and `.py` files are in the same directory is important. The folder structure is as follows:
- **`data and data_extra` folder**: store the instances presented in the paper.
- **`res_method_name` folders**: store results according to each method.

### Execution Results

After completing the executions, each `res_method_name` folder will contain 4 files for each requested execution. For example, in the `res_c2` folder, you will find:

1. **C2_1.txt**: pertains to the first requested execution and includes, for each instance, the following information:
   - Graph ID
   - Value \( d \) (maximum dislocation extent number of positions)
   - Solution obtained in the constructive phase
   - Solution obtained after applying local search
   - Total execution time

2. **C2_const_p_1.txt**: contains the vector with the positions of each node in each layer for the solution obtained in the constructive phase.

3. **C2_p_1.txt**: contains the vector with the positions of each node in each layer for the final solution.

4. **C2_iter_1.txt**: stores information gathered during each GRASP iteration, including:
   - Graph ID
   - Value \( d \)
   - Randomly sampled GRASP parameter \( φ \) in the interval [0,1]
   - GRASP iteration number
   - Initial solution obtained in the constructive phase
   - Final solution obtained after the local search
   - Total execution time

### Important Notes

- It is crucial that all mentioned folders and files are correctly organized within the same directory to ensure the proper functioning of the program.
- Verify the presence of all necessary files before initiating the execution.

Please note that in the results files, certain lines (corresponding to specific instances) may contain only a sequence of `-1`. It is crucial to emphasize that the number of incremental nodes in each layer must be greater than or equal to the \(d\) parameter. According to this criterion, these instances are the ones that did not meet the imposed requirement.
