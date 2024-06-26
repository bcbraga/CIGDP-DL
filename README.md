# CIGDP-DL
Constrained Incremental Graph Drawing Problem/Embeddings
## Operation of the Code

### Input Parameters

The program requests three input parameters from the user:
1. **Total number of executions**
2. **Total number of GRASP iterations**
3. **Complementary stopping criterion \( η_max \)**: terminates the code execution if \( η_max \) consecutive GRASP iterations occur without improvements.

An example of input parameters can be found in the `input.txt` file. If preferred, these parameters can be read directly from this file instead of being entered manually.

### Folder and File Structure

For the code execution, ensuring that the `res`, `data` folders, and `.py` files are in the same directory is important. The folder structure is as follows:
- **`data` folder**: stores the instances presented in the paper.
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
   - Randomly sampled GRASP parameter \( \phi \) in the interval [0,1]
   - GRASP iteration number
   - Initial solution obtained in the constructive phase
   - Final solution obtained after the local search
   - Total execution time

### Important Notes

- It is crucial that all mentioned folders and files are correctly organized within the same directory to ensure the proper functioning of the program.
- Verify the presence of all necessary files before initiating the execution.

Please note that in the results files, certain lines (corresponding to specific instances) may contain only a sequence of `-1`. It is crucial to emphasize that the number of incremental nodes in each layer must be greater than or equal to the \(d\) parameter. According to this criterion, these instances are the ones that did not meet the imposed requirement.
