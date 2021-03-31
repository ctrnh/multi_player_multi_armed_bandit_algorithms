# multi_player_multi_armed_bandit_algorithms

This repository allows to benchmark all state-of-the-art Multi Player Multi-Armed Bandit algorithms.
Implemented algorithms:
- **Cooperative and Stochastic Multi-Player Multi-Armed Bandit: Optimal Regret With Neither Communication Nor Collisions**, Sébastien Bubeck, Thomas Budzinski, Mark Sellke 
- SIC-MMAB, SIC-MMAB2 and DYN-MMAB algorithms from **SIC-MMAB: Synchronisation Involves Communication in Multiplayer Multi-Armed Bandits**, Etienne Boursier, Vianney Perchet
- EC-SIC from **Decentralized Multi-player Multi-armed Bandits with No Collision Information**, Chengshuai Shi, Wei Xiong, Cong Shen, Jing Yang
- First and Second algorithm from **Multiplayer bandits without observing collision information**, Gabor Lugosi, Abbas Mehrabian
- MCTopM, SelfishUCB from **Multi-Player Bandits Revisited**, Lilian Besson, Emilie Kaufmann
- Musical Chairs from **Multi-Player Bandits -- a Musical Chairs Approach**, Jonathan Rosenski, Ohad Shamir, Liran Szlak
- Randomized SelfishUCB from **A High Performance, Low Complexity Algorithm for Multi-Player Bandits Without Collision Sensing Information**, Cindy Trinh, Richard Combes


This is the code attached to the following paper:
***A High Performance, Low Complexity Algorithm for Multi-Player Bandits Without Collision Sensing Information***, *Cindy Trinh, Richard Combes*. (https://arxiv.org/abs/2102.10200)


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
