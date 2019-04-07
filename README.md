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
1. Fix the bug that inventory is not empty in the end.
2. Change the constraints from inventory lengths to money.
