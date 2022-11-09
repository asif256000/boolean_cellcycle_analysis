# NOTES

## Basic rules to check stages of Cell Cycle

1. START of cell division (end of stationary G1) requires Cln3
2. G1 requires Cln3 to be deactivated and MBF, SBF to be activated
3. G1 to S requires activation of Clb5,6
4. G2 requires activation of Clb1,2
5. Entering M phase is indicated by activation of Cdc20, 14
6. Exit from M to G1 phase is indicated by deactivation of the same Cyclins

## Iteration 0: Reproduce `Table 2` of the paper `The yeast cell-cycle network is robustly designed`

1. Consider the directed graph of cyclin network given in Fig 1.B of the paper.
2. Design an algorithm based on the formula given in the paper to get $S_{i}(t+1)$ based on the value of $\sum_j a_j S_j(t)$
3. Try to reproduce `Table 2` for 15 iterations (no stop condition so far, hence need to use hardcoded iteration value)
4. Also use self-degradation loops for the cyclins where it is required.
5. Reproduce the table as closely as possible.

## Iteration 1: Generate scores for different starting state for a fully connected graph

1. Generate a fully connected graph (all nodes connected to all other nodes with green edges).
2. Generate all possible ($2^{11}$) starting states for the cyclins.
3. Pass every state through the same algorithm as `iteration 0` for 50 iterations.
4. Generate a score for each starting state and store the score.
5. Use `Manhattan Distance` for calculating the score, i.e for every cyclin, take absolute difference between every final state of the cyclin and the expected final state of the cyclin. Sum of all such differences is the final score for the starting state.

## Upcoming tasks

1. Hardcode self-degradation loop of Swi5.
2. Create a score for various graphs and starting states:
   1. Generate a fully connected graph with edges = 1
   2. We can have $2^{11}$ different starting states
   3. Run every starting state with a loop for size 50
   4. Take the final condition after 50th iteration
   5. Score the iteration by subtracting final condition to expected condition and squaring the value to find distance
   6. The summation of all distances for all cyclins is the final score for that starting state
   7. Record scores for all starting states
