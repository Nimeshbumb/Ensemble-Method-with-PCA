Mtrain=csvread('file.csv',1,1);     //Reading Training Data
Rtest=csvread('test_data.csv');     //Reading Testing Data
Mtrain1=Mtrain(1:end,1:256);
Ytrain=Mtrain(1:end,257);
Rtest1=Rtest(1:end,1:256);
Xtrain=Mtrain1-repmat(mean(Mtrain1,1),size(Mtrain1,1),1);   //Calculating Empirical Mean of Training Data
Xtest=Rtest1-repmat(mean(Mtrain1,1),size(Rtest1,1),1);      //Calculating Empirical Mean of Testing Data

[coeff scores variance]=princomp(Xtrain);

new_variables = scores(:,1:13);         /* Applying PCA on Training Data after Cross Validation by choosing 13
                                        top features*/

[coefs,scores1,variances] = princomp(Xtrain,'econ');

testScores = Xtest*coefs(:,1:13);       //Applying PCA on Testing Data


dlmwrite('PCATraindata.csv',new_variables);

dlmwrite('PCATestdata.csv',testScores);