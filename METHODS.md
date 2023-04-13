# METHODS

We started with boolean cell cycle model of budding yeast from the [paper](https://www.pnas.org/doi/10.1073/pnas.0305937101) authored by Li, Long, Lu, Ouyang, Tang et all.

We developed our module to replicate the results of the paper, specifically the state path each start states go through (`Table 2` in the paper) and count the number of starting states that merge to each unique final state ((`Table 1` in the paper)).

The paper implements synchronous update scheme for the cell cycle progression, which is not very natural way cell cycle progresses. In synchronous cell cycle progression, all the proteins involved in the cell cycle go through the reaction at the same time instance based on their states in the .
