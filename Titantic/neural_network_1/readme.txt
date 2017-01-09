01-08-17:
This is my first crack at a neural network using sklearn.

I tried one with 10x10 and one with 20x20, I forgot to record the kaggle test results fo the 10x10, but the 20x20 results were 74.641, i think my best (76.077) was a logreg.

I resubmitted the 10x10 and the score from kaggle was 73.684

All the above was with alpha = 1e-5, im going to try regularizing the 20x20
with alpha == 15 I got .77033 (this beat the benchmark)
alpha == 20 I got .75120
alpha == 14 got .74641
