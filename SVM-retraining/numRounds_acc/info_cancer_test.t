Remarks time meassurements queries per round: cancer data set

There is no obvious trend or difference for larger/smaller amount 
of queries in the accuracy of the model. The error on a uniform distribution
of points is also smaller than the error on the training data. This is probably
because the space is mainly of class 1 and the ration class0/class1 is larger
for the training set than for the validation set. 

Conclusion:
Since there is no obvious winner, both for the toy data set as for the cancer
data set and the computation time increases drastically for very few queries per round (more than 3h for the cancer data set with qprrnd = 10) I propose to (depending on the number of features) take as number of queries per round

#parameters * b/20

With this qprrnd we have to do 20 iterations to finish. 


approximate time spent in the simulation of the figure (omgekeerd evenredig met het qprrnd)

qprrnd = 10	4h
qprrnd = 20	2h
qprrnd = 32	1h 15min
qprrnd = 50	48min
qprrnd = 100	24min 
