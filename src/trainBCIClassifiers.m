function [models, accuracy, confusionMatrix, info] = trainBCIClassifiers(features, triggers, config)
    % TRAINBCICLASSIFIERS - Train multiple ML models for BCI classification
    %
    % Inputs:
    %   features - Feature matrix [windows x features]
    %   triggers - Event markers (same length as windows)
    %   config - Configuration structure
    %
    % Outputs:
    %   models - Trained models structure
    %   accuracy - Accuracy for each model
    %   confusionMatrix - Confusion matrix for best model
    %   info - Training information
    
    fprintf('   Training classifiers...\n');
    
    % Prepare data
    % Ensure features and triggers have same number of samples
    nWindows = size(features, 1);
    if length(triggers) > nWindows
        % Resample triggers to match feature windows
        windowCenters = round(linspace(1, length(triggers), nWindows));
        triggers = triggers(windowCenters);
    elseif length(triggers) < nWindows
        triggers = [triggers, zeros(1, nWindows - length(triggers))];
    end
    
    % Get valid samples (non-zero triggers)
    validIdx = triggers > 0;
    X = features(validIdx, :);
    y = triggers(validIdx)';
    
    % Remove NaN values
    nanRows = any(isnan(X), 2);
    X = X(~nanRows, :);
    y = y(~nanRows);
    
    if isempty(X) || length(unique(y)) < 2
        error('Insufficient valid data for training');
    end
    
    fprintf('   Training with %d samples, %d features, %d classes\n', ...
        size(X, 1), size(X, 2), length(unique(y)));
    
    % Split data (70% train, 30% test)
    cv = cvpartition(y, 'HoldOut', 0.3);
    X_train = X(cv.training, :);
    y_train = y(cv.training);
    X_test = X(cv.test, :);
    y_test = y(cv.test);
    
    % Initialize structures
    models = struct();
    accuracy = struct();
    
    %% 1. Support Vector Machine (SVM)
    fprintf('   Training SVM...\n');
    try
        t = templateSVM('KernelFunction', 'rbf', 'Standardize', true, 'BoxConstraint', 1);
        models.svm = fitcecoc(X_train, y_train, 'Learners', t, 'Coding', 'onevsone');
        y_pred_svm = predict(models.svm, X_test);
        accuracy.svm = sum(y_pred_svm == y_test) / length(y_test);
    catch
        fprintf('     SVM training failed\n');
        accuracy.svm = 0;
    end
    
    %% 2. Random Forest
    fprintf('   Training Random Forest...\n');
    try
        models.rf = TreeBagger(50, X_train, y_train, ...
                              'Method', 'classification', ...
                              'OOBPrediction', 'on', ...
                              'MinLeafSize', 5);
        y_pred_rf = predict(models.rf, X_test);
        y_pred_rf = str2double(y_pred_rf);
        accuracy.rf = sum(y_pred_rf == y_test) / length(y_test);
    catch
        fprintf('     Random Forest training failed\n');
        accuracy.rf = 0;
    end
    
    %% 3. Linear Discriminant Analysis (LDA)
    fprintf('   Training LDA...\n');
    try
        models.lda = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
        y_pred_lda = predict(models.lda, X_test);
        accuracy.lda = sum(y_pred_lda == y_test) / length(y_test);
    catch
        fprintf('     LDA training failed\n');
        accuracy.lda = 0;
    end
    
    %% 4. K-Nearest Neighbors (KNN)
    fprintf('   Training KNN...\n');
    try
        models.knn = fitcknn(X_train, y_train, 'NumNeighbors', 5);
        y_pred_knn = predict(models.knn, X_test);
        accuracy.knn = sum(y_pred_knn == y_test) / length(y_test);
    catch
        fprintf('     KNN training failed\n');
        accuracy.knn = 0;
    end
    
    %% 5. Ensemble (Majority Voting)
    fprintf('   Creating Ensemble...\n');
    try
        % Collect predictions from successful models
        predictions = [];
        modelNames = fieldnames(models);
        
        for m = 1:length(modelNames)
            model = models.(modelNames{m});
            if ~isempty(model)
                pred = predict(model, X_test);
                if isnumeric(pred)
                    predictions = [predictions, pred];
                else
                    predictions = [predictions, str2double(pred)];
                end
            end
        end
        
        if ~isempty(predictions)
            % Majority voting
            y_pred_ensemble = mode(predictions, 2);
            accuracy.ensemble = sum(y_pred_ensemble == y_test) / length(y_test);
            
            % Store ensemble model info
            models.ensemble = struct();
            models.ensemble.models = modelNames;
            models.ensemble.predictions = predictions;
        else
            accuracy.ensemble = 0;
        end
    catch
        accuracy.ensemble = 0;
    end
    
    %% Determine best model
    accValues = [];
    accNames = fieldnames(accuracy);
    
    for i = 1:length(accNames)
        accValues(end+1) = accuracy.(accNames{i});
    end
    
    [bestAcc, bestIdx] = max(accValues);
    bestModelName = accNames{bestIdx};
    
    % Create confusion matrix for best model
    switch bestModelName
        case 'svm'
            y_pred_best = y_pred_svm;
        case 'rf'
            y_pred_best = y_pred_rf;
        case 'lda'
            y_pred_best = y_pred_lda;
        case 'knn'
            y_pred_best = y_pred_knn;
        case 'ensemble'
            y_pred_best = y_pred_ensemble;
        otherwise
            y_pred_best = y_test; % Fallback
    end
    
    % Create confusion matrix
    uniqueClasses = unique([y_test; y_pred_best]);
    confusionMatrix = confusionmat(y_test, y_pred_best);
    
    % Ensure square matrix
    nClasses = max(length(uniqueClasses), size(confusionMatrix, 1));
    if size(confusionMatrix, 1) < nClasses
        confusionMatrix(end+1:nClasses, end+1:nClasses) = 0;
    end
    
    % Store additional info
    info = struct();
    info.nTrainingSamples = length(y_train);
    info.nTestSamples = length(y_test);
    info.nClasses = length(unique(y_train));
    info.bestModel = bestModelName;
    info.bestAccuracy = bestAcc;
    info.classDistribution = histcounts(y_train);
    
    % Display results
    fprintf('\n   Classification Results:\n');
    fprintf('   -------------------------\n');
    for i = 1:length(accNames)
        fprintf('   %-12s: %.2f%%\n', accNames{i}, accuracy.(accNames{i}) * 100);
    end
    fprintf('\n   Best model: %s (%.2f%%)\n', bestModelName, bestAcc * 100);
    
    % Store confusion matrix in accuracy struct
    accuracy.confusionMatrix = confusionMatrix;
end