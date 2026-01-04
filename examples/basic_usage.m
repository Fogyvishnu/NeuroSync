%% NEURONSYNC - Basic Usage Example
% This script demonstrates the basic usage of NeuroSync

clear; close all; clc;

fprintf('==========================================\n');
fprintf('NEURONSYNC - Basic Usage Example\n');
fprintf('==========================================\n\n');

%% 1. Run NeuroSync with default settings
fprintf('1. Running NeuroSync with simulated data...\n');
NeuroSync();  % Uses simulated data by default

pause(2);

%% 2. Run with sample data
fprintf('\n\n2. Running NeuroSync with sample data...\n');
NeuroSync('sample');

pause(2);

%% 3. Run with custom options
fprintf('\n\n3. Running NeuroSync with custom options...\n');

options = struct(...
    'subject', 1, ...
    'showDashboard', true, ...
    'runRealTime', false, ...  % Disable real-time for this example
    'saveResults', true, ...
    'verbose', true ...
);

NeuroSync('simulated', options);

%% 4. Direct function calls (advanced)
fprintf('\n\n4. Direct function calls...\n');

% Load data
[eegData, triggers, channelInfo, config] = loadEEGData('sample');

% Preprocess
[eegClean, artifacts] = preprocessEEG(eegData, config);

% Extract features
[features, featureNames] = extractEEGFeatures(eegClean, config);

% Train classifiers
[models, accuracy] = trainBCIClassifiers(features, triggers, config);

fprintf('\nDirect processing complete!\n');
fprintf('Final accuracy: %.2f%%\n', accuracy.ensemble * 100);

%% 5. Create custom visualization
fprintf('\n\n5. Creating custom visualization...\n');

figure('Position', [100, 100, 1200, 400]);

% Plot raw vs clean EEG
subplot(1, 3, 1);
plot(eegData(1, 1:1000));
hold on;
plot(eegClean(1, 1:1000), 'r');
title('Raw vs Clean EEG (Channel 1)');
legend('Raw', 'Clean');
xlabel('Samples');
ylabel('Amplitude (μV)');
grid on;

% Plot feature distribution
subplot(1, 3, 2);
histogram(features(:, 1), 50);
title('Distribution of Feature 1 (Mean)');
xlabel('Value');
ylabel('Frequency');
grid on;

% Plot class distribution
subplot(1, 3, 3);
classDist = histcounts(triggers(triggers > 0));
bar(classDist);
title('Class Distribution');
xlabel('Class');
ylabel('Count');
set(gca, 'XTickLabel', config.classes);
grid on;

%% Completion
fprintf('\n==========================================\n');
fprintf('Basic usage example complete!\n');
fprintf('Check the created figures for results.\n');
fprintf('==========================================\n');