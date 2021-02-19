import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


from tqdm import tqdm
from .algorithm import *
from .utils import *

class Ecsic(Algorithm):
    """
    EC-SIC algorithm
    Note:
     - In EC-SIC, players remain synchronized if they manage to communicate correctly. If one player happens to not receive the right statistics sent by the leader, they may be non synchronized. This happens with very low probability. Therefore, for simplicity, we "manually" synchronize players, and use assert to make sure that each player receives the right statistics (M, K_p, mu_hat etc), and therefore are indeed supposed to be synchronized "in reality" too.
     - We found that the code type repetition gave slightly better results than Hamming code, although this depends heavily on the self.Q parameter. We did not tune this parameter, but the difference is not significant in any case.
     - We set self.p = 5 as suggested in the paper, to enhance performances (longer exploration phase, smaller number of communication phases.)
    """
    def __init__(self, environment,
                mu_min=None,
                Delta=None,
                 code_type="repetition"
                ):
        super().__init__(environment=environment)
        if self.env.M < self.env.K and self.env.mu[self.env.M-1] > self.env.mu[self.env.M]: # EC-SIC doesn't work in those cases
            self.mu_min = mu_min
            if mu_min is None:
                self.mu_min = self.env.mu[-1]

            if Delta is None:
                self.Delta = self.env.mu[self.env.M-1] - self.env.mu[self.env.M]

            self.epsilon = self.Delta/8

            self.T_c = int(np.ceil(np.log(self.env.T)/self.mu_min))

            self.Q = max(np.log2(1/(self.Delta/4-self.epsilon)), np.log2(self.env.K +1))
            #logger.info("Q : ",self.Q)
            self.code_type=code_type


            if self.code_type == "repetition":
                self.Q = int(np.ceil(self.Q))
                self.A = int(np.ceil(np.log(self.Q*self.env.T)/self.mu_min))
                N_code = self.Q * self.A


            elif self.code_type == "hamming":
                self.A = int(np.ceil(0.5*np.log(7*self.Q*self.env.T/8)/self.mu_min))
                self.Q = int(self.Q)

                self.Q = self.Q + (4 - self.Q%4) # Q est alors divisble par 4
                N_code = 7*self.Q/4 * self.A

                self.G_ham = np.array([[1,1,0,1],
                                       [1,0,1,1],
                                       [1,0,0,0],
                                       [0,1,1,1],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]])

                self.H_ham = np.array([[1,0,1,0,1,0,1],
                                       [0,1,1,0,0,1,1],
                                       [0,0,0,1,1,1,1]])

                self.R_ham = np.array([[0,0,1,0,0,0,0],
                                       [0,0,0,0,1,0,0],
                                       [0,0,0,0,0,1,0],
                                       [0,0,0,0,0,0,1]])

            elif self.code_type == "flip":
                self.A = int(np.ceil(np.log(self.Q*self.env.T/2)/self.mu_min))
                N_code = self.Q * self.A_flip
                raise NotImplementedError

            else:
                raise ValueError

            self.N_prime = max(self.Q/self.C_z(1-self.mu_min),
                                N_code)
            #logger.info(f"Initializing ECSIC. \n mu_min = {self.mu_min}, Delta = {self.Delta}, epsilon = {self.epsilon}, T_c = {self.T_c}, Q = {self.Q}, N_prime={self.N_prime}")


    def __str__(self):
        return f"EC-SIC"

    def reset(self):
        super().reset()

        # initialization
        self.M_p = np.ones((self.env.M,),dtype=int)
        self.internal_ranks = np.zeros((self.env.M,),dtype=np.int32)

        # explo comm exploit
        self.p = 5*np.ones((self.env.M,),dtype=int)
        self.fixed = np.ones((self.env.M,),dtype=int)*-1
        self.active_arms = [np.arange(self.env.K) for player in range(self.env.M)]

        # Exploration
        self.T_p = np.zeros((self.env.M, self.env.K)) # nb pulls de chaque bras pdt explo phase
        self.small_s = np.zeros((self.env.M, self.env.K)) # success pendant explo phase
        self.mu_hat = np.zeros((self.env.M, self.env.K)) # mu_hat estimé de ttes les explo phases

    def C_z(self,q):
        return np.log2(1 + (1-q) * q**(q/(1-q)))

    def check_M_K_p_common(self):
        # all players have same nb of active players?
        self.common_M_p = self.M_p[self.leader]
        assert (self.M_p[self.players_ord[:self.common_M_p]] == self.common_M_p ).all(), self.M_p

        # all active players have same set of active_arms?
        self.common_active_arms = self.active_arms[self.leader]
        self.common_p = self.p[self.leader]
        self.K_p = len(self.common_active_arms)

        for i in range(self.common_M_p):
            player = self.players_ord[i]
            assert (self.active_arms[player] == self.common_active_arms).all()
            assert self.p[player] == self.common_p


    def init_musical_chairs(self, MC_length):
        ## Musical chair of initialization phase
        #logger.info(f"****** Starting initialization phase... Musical chairs for {MC_length} steps")
        arms_t = -1*np.ones((self.env.M,))
        ext_rank = -1*np.ones((self.env.M,),dtype=int) #external rank
        for t_MC in range(MC_length):#while self.t < MC_length:
            for player in range(self.env.M):
                if ext_rank[player] == -1:
                    if self.t > 0 and rewards_t[player] >= 1:
                        ext_rank[player] = arms_t[player]
                        logger.debug(f" t={self.t}, MC: player {player} has fixed on {arms_t[player]}")
                    else:
                        arms_t[player] = np.random.choice(self.env.K)
                        #logger.debug(f" t={self.t}, MC: player {player} not fixz")
                else:
                    arms_t[player] = ext_rank[player]
                    #logger.debug(f" t={self.t}, MC: player {player} not fixz")

            # Pull arms
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            logger.debug(f" t={self.t}, MC: draw arms_t {arms_t}, rewards_t: {rewards_t}")
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1

        #logger.info(f"\n******** t = {self.t},  MC finished,  pulls: \n {self.pulls} ")
        return ext_rank

    def estimate_M_no_sensing(self, ext_rank):
        """
        update self.internal_ranks and self.M_p
        """

        pis = ext_rank.copy()
        for n in range(2*self.env.K):
            r = np.zeros((self.env.M,))
            for player in range(self.env.M):
                if n >= 2 * ext_rank[player]:
                    pis[player] = (pis[player] + 1)%self.env.K


            for t_estM in range(self.T_c):
                arms_t = pis.astype(int)
                rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
                self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
                r += rewards_t
                self.t += 1

            for player in range(self.env.M):
                if r[player] == 0:
                    if n < 2*ext_rank[player]:
                        self.internal_ranks[player] += 1
                    self.M_p[player] += 1

    def quantize(self, mu_hat):
        for k in range(2**self.Q):
            diff = (k+1)/(2**self.Q+1) - mu_hat
            if diff > 0.5/(2**self.Q+1):
                return max(0, k-1)
            if diff >= 0:
                return k
        return 2**self.Q-1

    def unquantize(self, k):
        return (k+1)/(2**self.Q+1)

    def encode(self, k):
        """
        Args:
            - k (int)
                integer to encode

        Return:
            - msg (list of binary integers, e.g. [0,0,0,1,1,1,0,0,0,])
                encoded k to send (sender draws receiver's arm to induce collision if they want to send 1).
                if code_type == "repetition": msg has length self.A*self.Q
                if code_type == "hamming": msg has length (Q/4)*7*self.A
        """
        bin_k = bin(k)[2:]
        if len(bin_k) < self.Q:
            bin_k = "0"*(self.Q - len(bin_k)) + bin_k
        msg = []
        if self.code_type == "repetition":
            for c in bin_k:
                msg = msg + [int(c)]*self.A

        elif self.code_type == "hamming":
            for i in range(self.Q//4): # if hamming, Q should be multiple of 4
                four_bits = np.zeros((4,))
                for j in range(4):
                    four_bits[j] = bin_k[i*4 + j]
                seven_bits_code = self.G_ham.dot(four_bits)%2
                for bit in seven_bits_code:
                    msg = msg + [int(bit)]*self.A
        else:
            raise ValueError
        return msg

    def decode(self, msg):
        """
        Args:
            - msg (list of binary integers)
                list of rewards received by the receiver during transmission.

        Return:
            - decoded_msg : (int)
                in [0, 2**Q-1]
        """
        decoded_msg = 0
        if self.code_type == "repetition":
            for k_bit in range(self.Q-1, -1, -1):
                one_or_zero = 1
                for rep in range(self.A):
                    if msg[(self.Q-1-k_bit)*self.A + rep] == 1:
                        one_or_zero = 0
                        break
                decoded_msg += one_or_zero*2**(k_bit)
        elif self.code_type == "hamming":
            decoded_binary_msg = []
            seven_bits = []
            for i in range(len(msg)//self.A):
                one_or_zero = 1
                for j in range(self.A):
                    if msg[i*self.A + j] == 1:
                        one_or_zero = 0
                        break
                seven_bits.append(one_or_zero)
                if len(seven_bits) == 7:
                    decoded_binary_msg += list(self.R_ham.dot(seven_bits)) # append 4 bits
                    seven_bits = []
            for k_bit in range(self.Q):
                decoded_msg += decoded_binary_msg[k_bit]*2**(self.Q-1-k_bit)
        return decoded_msg

    def send(self, sender, receiver, msg, arms_t):
        """
        !! arms_t must be filled for players all non sender
        msg: encoded message (list of binary numbers)
        """
        code_received = []
        for c in msg:
            if c == 1: # to send 1: sender draws same arm as receiver
                arms_t[sender] = self.common_active_arms[self.internal_ranks[receiver]]
            else:
                arms_t[sender] = self.common_active_arms[self.internal_ranks[sender]]
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1
            ##logger.info("arms_t:", arms_t, "\n rewards:", rewards_t,"\n")
            code_received.append(int(rewards_t[receiver]))
        return code_received

    def compute_rej_acc_sets(self, mu_hat_leader):
        """
        mu_hat_leader : (K)

        """
        rej = []
        acc = []
        B = np.sqrt((2*np.log(self.env.T))/self.T_p[self.leader,self.common_active_arms[0]]) + self.Delta/4 - self.epsilon
       # #logger.info(f"\n..... p = {self.common_p}, t = {self.t}, mu_hat_leader: \n {mu_hat_leader} \n ub: \n {mu_hat_leader +B}, \n lb \n {mu_hat_leader-B} ")

        for arm_k in self.common_active_arms:
            ub_k = mu_hat_leader[arm_k] + B
            lb_k = mu_hat_leader[arm_k] - B
            count_rej = np.sum(mu_hat_leader[self.common_active_arms] - B >= ub_k)
            count_acc = np.sum(lb_k >= mu_hat_leader[self.common_active_arms] + B)

            logger.debug(f"count_acc = {count_acc}, count_rej = {count_rej}")

            if count_rej >= self.common_M_p:
                rej.append(arm_k)
            if count_acc >= len(self.common_active_arms) - self.common_M_p:
                acc.append(arm_k)

        return rej, acc


    def run(self):
        #logger.info(f"Running {str(self)}, T_c = {self.T_c}, mu_min = {self.mu_min}")

        ##### INITIALIZATION PHASE #####
        # MC
        mc_ext_rank = self.init_musical_chairs(MC_length=self.env.K*self.T_c)
        #logger.info(f"...... t = {self.t} : Finished initialization MC: {mc_ext_rank}")
        # t = self.env.K*self.T_c

        # Estimating M
        self.estimate_M_no_sensing(mc_ext_rank) # lasts 2*K*T_c
        assert self.t == self.env.K*self.T_c +  2*self.env.K*self.T_c
        #logger.info(f"...... t = {self.t} : Finished initialization Estimate_M: estimation M_p = {self.M_p}, internal ranks: {self.internal_ranks}")
        ##### END INITIALIZATION PHASE #####

        # check estimate_M
        assert len(np.unique(self.internal_ranks)) == self.env.M, self.internal_ranks
        assert self.M_p[0] == self.env.M

        # sorting by internal ranks:
        self.players_ord = np.argsort(self.internal_ranks)
        self.leader = self.players_ord[0] #idx of the leader
        self.check_M_K_p_common()
        ##### EXPLO-COMM-EXPLOIT #####
        #logger.info(f"internal ranks: {self.internal_ranks} \n players ord: {self.players_ord}")

        arms_t = -1*np.ones((self.env.M,), dtype=np.int32)

        while self.t < self.env.T:


            assert (self.fixed[self.players_ord[self.common_M_p:]] != -1).all(), self.fixed
            #arms_t[self.fixed != -1] = self.fixed[self.fixed != -1]
            fixed_players = self.players_ord[self.common_M_p:]
            active_players = self.players_ord[:self.common_M_p]
            arms_t[fixed_players] = self.fixed[fixed_players]
            if self.common_M_p == 0:
                rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
                self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
                self.t += 1
            else:
                # exploration
                pi = self.internal_ranks.copy()
                for t_explo in range(self.K_p * 2**self.common_p * int(np.ceil(np.log(self.env.T)))):
                    for player in active_players:
                        arms_t[player] = self.common_active_arms[pi[player]]
                        pi[player] = int(pi[player] + 1)%self.K_p
                    ##logger.info(f"\n..... p = {self.common_p}, t = {self.t}, arms_t = {arms_t}, active_players: {active_players}")
                    rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
                    self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
                    self.t += 1
                    self.T_p[np.arange(self.env.M), arms_t] += 1 # !! les players fixés aussi updatent
                    self.small_s[np.arange(self.env.M),arms_t] += rewards_t
                    ##logger.info(f"t = {self.t}, arms_t = {arms_t}, rewards_t = {rewards_t}, small_s = \n {self.small_s},T_p: \n {self.T_p}")

                self.mu_hat_explo = self.small_s/self.T_p
                ##logger.info(f"\n..... p = {self.common_p}, t = {self.t}, finished exploration. mu_hat:\n{self.mu_hat_explo}")

                # communication
                ## Transmitting mu_hats
                mu_hat_leader = np.zeros((self.common_M_p, self.env.K)) # row: active player but col: all arms
                mu_hat_leader[0,self.common_active_arms] = self.mu_hat_explo[self.leader,self.common_active_arms] # first line is leader
                arms_t[active_players] = self.common_active_arms[:self.common_M_p]# = self.common_active_arms[self.internal_ranks[active_players]]
                # for arms in self.common_M_p: : they are already fixed
                for i in range(1, self.common_M_p):
                    sender = self.players_ord[i]
                    for k in self.common_active_arms:
                        # For each arm, send Q bits, ie Q*self.A_rep draws
                        nb_to_send = self.quantize(self.mu_hat_explo[sender,k])

                        code_received = self.send(sender=sender,
                                                  receiver=self.leader,
                                                  msg=self.encode(nb_to_send),
                                                  arms_t=arms_t.copy())
                        ##logger.info(f"\n ... t = {self.t}, sender is player {sender}, receiver is leader {self.leader}: \n mu of arm {k} to send: {self.mu_hat_explo[sender,k]}, quantized: {nb_to_send}, msg = \n {self.encode(nb_to_send)}, code received: \n {code_received}\n")
                        nb_received = self.decode(code_received)
                        mu_hat_leader[i, k] = self.unquantize(nb_received)
                        assert nb_to_send == nb_received, str(nb_to_send) + "!=" +str(nb_received)
                        assert  abs(mu_hat_leader[i,k] - self.mu_hat_explo[sender,k]) <= 1/2**self.Q

                final_mu_hat_leader = np.sum(mu_hat_leader*self.T_p[self.leader,self.common_active_arms[0]],axis=0) /(self.common_M_p* self.T_p[self.leader,self.common_active_arms[0]])

                ##logger.info(f"\n..... p = {self.common_p}, t = {self.t}, Finished transmission mu_hat explo, mu_hat received: \n {mu_hat_leader}, \n mu_hat final = \n {final_mu_hat_leader}")

                rej, acc = self.compute_rej_acc_sets(mu_hat_leader=final_mu_hat_leader)
                #if len(rej) +len(acc) != 0:
                    #logger.info(f"\n..... p = {self.common_p}, t = {self.t}, \n Rejecting arms: {rej},\n Accepting arms: {acc}")
                encoded_nb_rej = self.encode(len(rej))
                encoded_nb_acc = self.encode(len(acc))

                ## Send nb rej, nb acc
                arms_t[active_players] = self.common_active_arms[:self.common_M_p]
                for i in range(1, self.common_M_p):
                    receiver = self.players_ord[i]
                    code_received_rej = self.send(sender=self.leader,
                                                  receiver=receiver,
                                                  msg=encoded_nb_rej,
                                                  arms_t=arms_t.copy()
                                                  )
                    code_received_acc = self.send(sender=self.leader,
                                                  receiver=receiver,
                                                  msg=encoded_nb_acc,
                                                  arms_t=arms_t.copy())

                    decoded_rej = self.decode(code_received_rej)
                    decoded_acc = self.decode(code_received_acc)

                    assert decoded_rej == len(rej), "nb of rej transmission failed"
                    assert decoded_acc == len(acc), "nb of acc transmission failed"


                encoded_rej_arms = [self.encode(k) for k in rej]
                encoded_acc_arms = [self.encode(k) for k in acc]
                accepted_arms = [[] for i in range(self.common_M_p)]
                rejected_arms = [[] for i in range(self.common_M_p)]

                accepted_arms[0] = acc
                rejected_arms[0] = rej

                arms_t[active_players] = self.common_active_arms[:self.common_M_p]
                for i in range(1, self.common_M_p):
                    receiver = self.players_ord[i]
                    for k in range(len(rej)):
                        code_received_rej_arm = self.send(sender=self.leader,
                                                          receiver=receiver,
                                                          msg=encoded_rej_arms[k],
                                                         arms_t=arms_t.copy())
                        decoded_arm_k = self.decode(code_received_rej_arm)
                        rejected_arms[i].append(decoded_arm_k)
                        assert decoded_arm_k == rej[k], f"rejected arm {rej[k]} not well decoded ({decoded_arm_k}) receiver:{receiver}"

                    for k in range(len(acc)):
                        code_received_acc_arm = self.send(sender=self.leader,
                                                          receiver=receiver,
                                                          msg=encoded_acc_arms[k],
                                                          arms_t=arms_t.copy())
                        decoded_arm_k = self.decode(code_received_acc_arm)
                        accepted_arms[i].append(decoded_arm_k)
                        assert decoded_arm_k == acc[k], f"accepted arm {acc[k]} not well decoded ({decoded_arm_k}),receiver:{receiver}"

                for j in range(self.common_M_p):
                    player = self.players_ord[j]
                    if len(acc) >= self.common_M_p - j:
                        self.fixed[player] = accepted_arms[j][self.common_M_p - j-1]

                        #logger.info(f"\n..... p = {self.common_p}, t = {self.t}, player {player} now fixed on arm {self.fixed[player]}")

                    self.M_p[player] -= len(acc)
                    self.active_arms[player] = np.array(np.setdiff1d(self.active_arms[player],
                                                               np.unique(accepted_arms[j]+rejected_arms[j])),
                                                               dtype=int)
                    self.p[player] += 1
                self.check_M_K_p_common()
                ##logger.info(f"common_M_p= {self.common_M_p}, active_arms:{self.common_active_arms}")
