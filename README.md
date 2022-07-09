# LTH-Reinit
🎟 Reinitializing pruned weights from the winning ticket

## Status
Quick idea I had regarding the signifigance of the initialization of pruned weights. After each step of IMP the "winning" weights have their initializations restored while the pruned weights are masked out. I tried reinitializng the pruned weights to new weights but this had no effect on model performance.

## Credit
Majority of code comes from Jonathan Frankle's open-lth repo: https://github.com/facebookresearch/open_lth
