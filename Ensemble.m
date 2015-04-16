Mtrain=csvread('file.csv',1,1);
Mtrain1=csvread('PCATraindata.csv');
Rtest=csvread('PCATestdata.csv');
Mtrainpca=Mtrain1(1:end,1:end);
Ytrain=Mtrain(1:end,257);
Rtestpca=Rtest(1:end,1:end);

%cvpart = cvpartition(Ytrain,'holdout',0.3);
%Xtrainpca = Mtrainpca(training(cvpart),:);
%Ytrainpca = Ytrain(training(cvpart),:);
%Xtestpca = Mtrainpca(test(cvpart),:);
%Ytestpca = Ytrain(test(cvpart),:);


%BaggedEnsemble=TreeBagger(60,Mtrainpca,Ytrain,'OOBPred','On'); //Random Forest(RF)

bag = fitensemble(Mtrainpca,Ytrain,'AdaBoostM2',100,'Tree','Type','Classification'); // Boosting

%cv = fitensemble(Mtrainpca,Ytrain,'Bag',60,'Tree','type','classification','kfold',10); //K-Fold CV

[predtest scores] = bag.predict(Rtestpca);
%Lgl=loss(BaggedEnsemble,Xtestpca,Ytestpca); // Misclassification Error


dlmwrite('PredictionBoosting.csv',scores);  // Printing Prediction for various Methods(For eg here: Boosting)



%oobErrorBaggedEnsemble = oobError(BaggedEnsemble);     //Out of Bag Error for RF
%plot(oobErrorBaggedEnsemble)
%xlabel 'Number of grown trees';
%ylabel 'Out-of-bag classification error';



%plot(loss(bag,Xtestpca,Ytestpca,'mode','cumulative')); //Plotting Independent Test Set Error and K-Fold CV Error
%hold on;
%plot(kfoldLoss(cv,'mode','cumulative'),'r');
%hold off;
%xlabel('Number of Subspaces');
%ylabel('Classification error');
%legend('Test','Cross-validation','Location','NE');

