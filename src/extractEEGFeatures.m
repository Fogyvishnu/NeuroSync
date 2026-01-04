function [features, featureNames, config] = extractEEGFeatures(eegData, datasetConfig)
    % EXTRACTEEGFEATURES - Extract comprehensive EEG features
    %
    % Inputs:
    %   eegData - Preprocessed EEG data [channels x samples]
    %   datasetConfig - Dataset configuration
    %
    % Outputs:
    %   features - Feature matrix [windows x features]
    %   featureNames - Cell array of feature names
    %   config - Feature extraction configuration
    
    fs = datasetConfig.sampling_rate;
    [nChannels, nSamples] = size(eegData);
    
    % Configuration
    config = struct();
    config.windowSize = 2 * fs;     % 2-second windows
    config.overlap = 1 * fs;        % 50% overlap
    config.nFeaturesPerChannel = 15;
    
    % Create window indices
    windows = 1:(config.windowSize - config.overlap):(nSamples - config.windowSize + 1);
    nWindows = length(windows);
    
    fprintf('   Extracting features from %d windows...\n', nWindows);
    
    % Initialize feature matrix
    totalFeatures = nChannels * config.nFeaturesPerChannel;
    features = zeros(nWindows, totalFeatures);
    featureNames = {};
    
    % Process each window
    for w = 1:nWindows
        startIdx = windows(w);
        endIdx = startIdx + config.windowSize - 1;
        windowData = eegData(:, startIdx:endIdx);
        
        featureVector = [];
        
        for ch = 1:nChannels
            signal = windowData(ch, :);
            
            % Extract features for this channel
            channelFeatures = extractChannelFeatures(signal, fs);
            featureVector = [featureVector, channelFeatures];
            
            % Create feature names (first window only)
            if w == 1
                baseNames = getFeatureBaseNames();
                for f = 1:length(baseNames)
                    featureNames{end+1} = sprintf('Ch%02d_%s', ch, baseNames{f});
                end
            end
        end
        
        features(w, :) = featureVector;
        
        % Progress indicator
        if mod(w, 10) == 0
            fprintf('     Processed %d/%d windows\n', w, nWindows);
        end
    end
    
    fprintf('   Extracted %d features\n', size(features, 2));
end

function features = extractChannelFeatures(signal, fs)
    % Extract features for a single channel
    
    features = zeros(1, 15);
    
    %% Time Domain Features (1-7)
    % 1. Mean
    features(1) = mean(signal);
    
    % 2. Variance
    features(2) = var(signal);
    
    % 3. Skewness
    features(3) = skewness(signal);
    
    % 4. Kurtosis
    features(4) = kurtosis(signal);
    
    % 5-7. Hjorth parameters
    [activity, mobility, complexity] = hjorth_parameters(signal);
    features(5) = activity;
    features(6) = mobility;
    features(7) = complexity;
    
    %% Frequency Domain Features (8-15)
    % Compute power spectral density
    [pxx, f] = pwelch(signal, [], [], [], fs);
    
    % 8. Total power
    features(8) = sum(pxx);
    
    % 9-13. Band powers
    deltaIdx = f >= 1 & f <= 4;
    thetaIdx = f >= 4 & f <= 8;
    alphaIdx = f >= 8 & f <= 13;
    betaIdx = f >= 13 & f <= 30;
    gammaIdx = f >= 30 & f <= 45;
    
    features(9) = sum(pxx(deltaIdx));   % Delta power
    features(10) = sum(pxx(thetaIdx));  % Theta power
    features(11) = sum(pxx(alphaIdx));  % Alpha power
    features(12) = sum(pxx(betaIdx));   % Beta power
    features(13) = sum(pxx(gammaIdx));  % Gamma power
    
    % 14. Spectral edge frequency (95%)
    cumPower = cumsum(pxx);
    sef95Idx = find(cumPower >= 0.95*cumPower(end), 1);
    if ~isempty(sef95Idx)
        features(14) = f(sef95Idx);
    else
        features(14) = 0;
    end
    
    % 15. Mean frequency
    if sum(pxx) > 0
        features(15) = sum(f .* pxx) / sum(pxx);
    else
        features(15) = 0;
    end
end

function baseNames = getFeatureBaseNames()
    % Return base names for features
    baseNames = {
        'Mean', 'Variance', 'Skewness', 'Kurtosis', ...
        'HjorthActivity', 'HjorthMobility', 'HjorthComplexity', ...
        'TotalPower', 'Delta', 'Theta', 'Alpha', 'Beta', ...
        'Gamma', 'SEF95', 'MeanFreq'
    };
end