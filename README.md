# Requirements

- Python3

- numpy, matplotlib

- (Optional) To improve speed, compile the cythonized version of the compute of KL-UCB index:
 ```
 cd multi_player_multi_armed_bandits/algorithms/cklucb
 python setup.py build_ext --inplace
 ```

# How to run

To reproduce the results of any section of the paper, run the corresponding script.
For example, to reproduce the figures from Section 4.1, run:
```
cd multi_player_multi_armed_bandits
python paper_4_1_linearly_spaced_mu.py
```
This will save results into the folder `code/results` and `code/results_plots`
