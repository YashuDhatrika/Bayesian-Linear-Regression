Assignment4: 


The pp2code.py file contains broadly 4 parts, one for each task. In totality the code picks required CSV file from the directory of the code file. The path isn’t hard coded, but the files need to be placed in code file directory.
And compatible on python 3+ version.

PART 0: Import and Storage of CSV files

All the data in the CSV file is converted to ndarry using the function genfromtxt from numpy library. And finally a dictionary is made where keys are file name and values are the ndarry.


PART 1 : Implementation of Code for Task 1

1. Intitally the object assignment is done for all the 5 dataset, each dataset as 4 object train array, target values of train, test array and target values of test.
2. Then Function finding_w is created which calculates array w for each lambda(regularization parameter).
3. Then function MSE is created which calculates the the MSE for each lamba.

Output:
It pops the graphs (MSE for both Training and Testing dataset as function of regularization parameter) for all the 5 dataset 



PART 2 : Implementation of Code for Task 2

For this task we are creating 3 functions:
	a)randomsample(): This function helps to generate the sample of given size
	b)q2_finding_w(): This function creates the W for 3 values of lamda ( very low,just correct and high value)
	c)q2_MSE(): This function calculate the MSE for the testing data which is trained on a given training dataset for 3 values of lamba
	d) Lastly, then calculates the mean of Mean square error of the samples generated from each training set of
	#sample size ranging from 10 to 800,  as sample for each sample size is drawn 10 time, mean of these 10 MSE.
Output:

It pops the graphs of Mean of MSE as function of Sample size for each lamba value.

PART 3 : Implementation of Code for Task 3:

For this task we are creating 2 function:

	a) obtain_modelparameter: This function calculates the model parameter alpha and beta using an iterative approach
	b) MSE_trainset(): This function calculates the MSE using the MAP estimate approach for a given test dataset.

Output:
It displays the MSE for all the 5 dataset.
Observations for Task 3
MSE using MAP estimate for Dataset-10 Features and 100 Example: 3.7605553848377644
MSE using MAP estimate for Dataset-100 Features and 100 Example: 3.4203946090564004
MSE using MAP estimate for Dataset-100 Features and 1000 Example: 3.630970796811492
MSE using MAP estimate for Wine Dataset-11 Features and 342 Example: 0.565523748453752
MSE using MAP estimate for Crime Dataset-100 Features and 298 Example: 0.31623086725603267


PART 4 : Implementation of Code for Task 4:

1. Intitally the object assignment is done for all the 2 dataset (f3,f5), each dataset as 4 object train array, target values of train, test array and target values of test.
2. create_example(): This function modifies the data in the training and testing set to a mentioned degree (1,x,x^degree)
3. q4_obtain_modelparameter():This function calculates the model parameter alpha and beta using an iterative approach.
4.Evidence():This function calculates the log of evidence for a given training set.
5.q4_MSE_MAP: This function calculates the MSE using the MAP estimate approach for a given test dataset.
6.q4_MSE_NR:his function calculates the MSE using the non regularized approach for a given test dataset.

Output for Task4:
1. It displays the 3 dictionaries where keys for all of them is degree and the values are MSE using MAP approach, MSE using the Non Regularization Method and the log of Evidence.
2. It also pops the graps of the above mentioned attribute for both of the datasets f3 and f5.


**Please note that after every task respective graphs pops up, so please close them to get the code executed further.




