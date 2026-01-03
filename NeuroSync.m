function NeuroSync(datasetType, options)
    % NEURONSYNC - EEG Brain-Computer Interface System
    % 
    % Syntax:
    %   NeuroSync()
    %   NeuroSync(datasetType)
    %   NeuroSync(datasetType, options)
    %
    % Inputs:
    %   datasetType - 'simulated', 'bci_iv_2a', 'physionet', 'deap', 'sample'
    %   options - Structure with configuration parameters
    %
    % Examples:
    %   NeuroSync()                           % Uses simulated data
    %   NeuroSync('sample')                   % Uses sample data
    %   NeuroSync('bci_iv_2a', struct('subject', 1))
    %
    % Author: Vishnu Prashanth M
    % Date: 3/1/2026
    
    fprintf('==================================================\n');
    fprintf('🧠 NEUROSYNC - EEG Brain-Computer Interface System\n');
    fprintf('==================================================\n\n');
    
    %% Default configuration
    if nargin < 1
        datasetType = 'simulated';
    end
    if nargin < 2
        options = struct();
    end
    
    % Merge with default options
    defaultOptions = struct(...
        'subject', 1, ...
        'session', 1, ...
        'showDashboard', true, ...
        'runRealTime', true, ...
        'saveResults', false, ...
        'verbose', true ...
    );
    
    options = mergeStructs(defaultOptions, options);
    
    %% 1. LOAD EEG DATA
    fprintf('1. Loading EEG Dataset: %s\n', datasetType);
    [eegData, triggers, channelInfo, datasetConfig] = loadEEGData(datasetType, options);
    
    if options.verbose
        fprintf('   Data size: %d channels x %d samples\n', size(eegData, 1), size(eegData, 2));
        fprintf('   Sampling rate: %d Hz\n', datasetConfig.sampling_rate);
        fprintf('   Number of events: %d\n', sum(triggers > 0));
    end
    
    %% 2. PREPROCESSING
    fprintf('2. Preprocessing EEG Signals...\n');
    [eegClean, artifacts, preprocInfo] = preprocessEEG(eegData, datasetConfig);
    
    if options.verbose
        fprintf('   Artifacts removed: %.1f%%\n', artifacts.artifactPercentage);
        fprintf('   Bad channels: %d/%d\n', sum(artifacts.deadChannels), size(eegData, 1));
    end
    
    %% 3. FEATURE EXTRACTION
    fprintf('3. Extracting Features...\n');
    [features, featureNames, featureConfig] = extractEEGFeatures(eegClean, datasetConfig);
    
    if options.verbose
        fprintf('   Extracted %d features\n', size(features, 2));
        fprintf('   Feature window: %.1f seconds\n', featureConfig.windowSize / datasetConfig.sampling_rate);
    end
    
    %% 4. TRAIN CLASSIFIERS
    fprintf('4. Training Machine Learning Models...\n');
    [models, accuracy, confusionMatrix, trainingInfo] = trainBCIClassifiers(features, triggers, datasetConfig);
    
    % Display results
    displayClassificationResults(accuracy, confusionMatrix, datasetConfig);
    
    %% 5. CREATE VISUALIZATION DASHBOARD
    if options.showDashboard
        fprintf('5. Creating Visualization Dashboard...\n');
        createNeuroDashboard(eegClean, features, models, accuracy, triggers, datasetConfig);
    end
    
    %% 6. REAL-TIME BCI INTERFACE
    if options.runRealTime
        fprintf('6. Launching Real-time BCI Interface...\n');
        realTimeBCI(models, datasetConfig);
    end
    
    %% 7. SAVE RESULTS
    if options.saveResults
        fprintf('7. Saving Results...\n');
        saveResults(eegClean, features, models, accuracy, datasetConfig);
    end
    
    %% Completion
    fprintf('\n✅ NeuroSync Pipeline Complete!\n');
    fprintf('==================================================\n');
    
end

%% Helper Functions

function result = mergeStructs(default, custom)
    % Merge two structures, custom overwrites default
    result = default;
    if isempty(custom)
        return;
    end
    fields = fieldnames(custom);
    for i = 1:length(fields)
        result.(fields{i}) = custom.(fields{i});
    end
end

function displayClassificationResults(accuracy, confusionMatrix, config)
    fprintf('\n📊 CLASSIFICATION RESULTS:\n');
    fprintf('─────────────────────────────────────\n');
    
    modelNames = fieldnames(accuracy);
    for i = 1:length(modelNames)
        if strcmp(modelNames{i}, 'confusionMatrix')
            continue;
        end
        fprintf('   %-15s: %.2f%%\n', modelNames{i}, accuracy.(modelNames{i}) * 100);
    end
    
    % Find best model
    accValues = [];
    for i = 1:length(modelNames)
        if ~strcmp(modelNames{i}, 'confusionMatrix')
            accValues(end+1) = accuracy.(modelNames{i});
        end
    end
    [bestAcc, bestIdx] = max(accValues);
    fprintf('\n   Best Model: %s (%.2f%%)\n', modelNames{bestIdx}, bestAcc * 100);
    
    % Display confusion matrix if available
    if ~isempty(confusionMatrix)
        fprintf('\n   Confusion Matrix:\n');
        disp(confusionMatrix);
    end
end

function saveResults(eegData, features, models, accuracy, config)
    % Save results to file
    
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('results/reports/neurosync_results_%s.mat', timestamp);
    
    % Create results structure
    results = struct();
    results.timestamp = timestamp;
    results.eegData = eegData;
    results.features = features;
    results.models = models;
    results.accuracy = accuracy;
    results.config = config;
    
    % Ensure directory exists
    if ~exist('results/reports', 'dir')
        mkdir('results/reports');
    end
    
    % Save
    save(filename, 'results');
    fprintf('   Results saved to: %s\n', filename);
    
    % Also save as CSV for easy viewing
    csvFile = sprintf('results/reports/accuracy_%s.csv', timestamp);
    modelNames = fieldnames(accuracy);
    accValues = zeros(length(modelNames)-1, 1);
    validIdx = 1;
    
    for i = 1:length(modelNames)
        if ~strcmp(modelNames{i}, 'confusionMatrix')
            accValues(validIdx) = accuracy.(modelNames{i}) * 100;
            validIdx = validIdx + 1;
        end
    end
    
    T = table(modelNames(1:end-1), accValues, 'VariableNames', {'Model', 'Accuracy'});
    writetable(T, csvFile);
    fprintf('   Accuracy table saved to: %s\n', csvFile);
end