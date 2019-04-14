## Modifications

### 1. More state variables
Add:
* volatility
* volume divided by Sp’s volume

### 2. Trading cost
Add trading cost inside the reward updating function.


## To Run

`python3 train.py GOOG1718 10 1000` where 10 is the window length and 1000 is the number of epochs.

## Things to work on

1. Add SP500 returns, inventory(Cash)(total money left) as state variables.
2. Add some visualization.
3. Fix the bug that inventory is not empty in the end.
4. Change the constraints from inventory lengths to money.
5. Add some benchmarks, like buy and hold entil the end.
6. Sharpe ratios.
7. Record how many times it traded.
8. Run facebook, google, amazon, netflix, aaple. (faang).
9. Add PNL.
10. Plot every state variables and things to learn(returns).
11. Change reward functions.
12. How to setup validation? Is continuous learning possible? 


