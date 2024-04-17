function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)

% Returns the trained classifier and its accuracy. The following code recreates a classification model trained in a classification learner. you can use it
% The generated code automatically trains the same model on new data, or it learns how to train the model programmatically.
%
% input:
%       trainingData: A matrix with the same number of columns and data types as the matrix imported into the app.
%
% output:
%       trainedClassifier: A structure containing the trained classifier. This structure contains various information about the trained points
%       Fields for classifier information.
%
%       trainedClassifier.predictFcn: A function that predicts new data.
%
%       validationAccuracy: Double containing accuracy percentage. In the app, the Models pane displays each
%       Overall accuracy score for % models.
%
% Use this code to train a model on new data. To retrain the classifier, use the original data or new data as input parameters
% trainingData Call this function from the command line.
%
% For example, to retrain a classifier trained on the original dataset T, enter:
% [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To use the returned "trainedClassifier" to make predictions on new data T2, use
% yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a matrix containing only predictor columns for training. For details please enter:
% trainedClassifier.HowToPredict



% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3'});

predictorNames = {'column_1', 'column_2'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_3;
isCategoricalPredictor = [false, false];

% train classifier
% The following code specifies all classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 226);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]);

% Use the prediction function to create the result structure
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add fields to the result structure
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This structure is a trained model exported from Classification Learner R2021b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix \n \nX must contain exactly 2 columns because this model is trained with 2 predictors. \nX must contain only \npredictor columns in the exact same order and format as the training data. Do not include response columns or any columns that are not imported into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and responses
% The following code processes the data into a suitable shape for training the model.
%
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3'});

predictorNames = {'column_1', 'column_2'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_3;
isCategoricalPredictor = [false, false];

% Perform cross validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

% Calculate verification prediction
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Calculate verification accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
