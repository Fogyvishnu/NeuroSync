function createNeuroDashboard(eegData, features, models, accuracy, triggers, config)
    % CREATENEURODASHBOARD - Create comprehensive visualization dashboard
    %
    % Inputs:
    %   eegData - Preprocessed EEG data
    %   features - Extracted features
    %   models - Trained models
    %   accuracy - Model accuracies
    %   triggers - Event markers
    %   config - Configuration
    
    fprintf('   Creating visualization dashboard...\n');
    
    % Create figure
    fig = figure('Name', 'NeuroSync - Analysis Dashboard', ...
                 'NumberTitle', 'off', ...
                 'Position', [50, 50, 1400, 800], ...
                 'Color', [0.05, 0.05, 0.1]);
    
    %% 1. EEG Time Series
    subplot(3, 4, 1);
    plotEEGTimeSeries(eegData, triggers, config.sampling_rate);
    title('EEG Time Series', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% 2. Power Spectrum
    subplot(3, 4, 2);
    plotPowerSpectrum(eegData, config.sampling_rate);
    title('Power Spectrum', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% 3. Topographical Map
    subplot(3, 4, 3);
    plotTopographicalMap(eegData);
    title('Topographical Distribution', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% 4. Model Performance
    subplot(3, 4, 4);
    plotModelPerformance(accuracy);
    title('Model Performance', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% 5. Feature Importance
    subplot(3, 4, [5, 6]);
    plotFeatureImportance(features, models);
    title('Feature Importance', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% 6. Time-Frequency Analysis
    subplot(3, 4, [7, 8]);
    plotTimeFrequency(eegData(1, :), config.sampling_rate); % First channel only
    title('Time-Frequency Analysis', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% 7. Confusion Matrix
    subplot(3, 4, [9, 10]);
    if isfield(accuracy, 'confusionMatrix') && ~isempty(accuracy.confusionMatrix)
        plotConfusionMatrix(accuracy.confusionMatrix, config.classes);
        title('Confusion Matrix', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    else
        text(0.5, 0.5, 'No confusion matrix available', ...
             'HorizontalAlignment', 'center', 'Color', 'white');
        axis off;
    end
    
    %% 8. System Information
    subplot(3, 4, [11, 12]);
    plotSystemInfo(eegData, features, models, accuracy, config);
    title('System Information', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    %% Add title
    annotation('textbox', [0.3, 0.95, 0.4, 0.04], ...
               'String', '🧠 NEURONSYNC - EEG BCI ANALYSIS DASHBOARD', ...
               'HorizontalAlignment', 'center', ...
               'FontSize', 16, 'FontWeight', 'bold', ...
               'EdgeColor', 'none', 'BackgroundColor', [0.2, 0.2, 0.4], ...
               'Color', 'white');
    
    fprintf('   Dashboard created successfully\n');
end

function plotEEGTimeSeries(eegData, triggers, fs)
    % Plot EEG time series with event markers
    
    [nChannels, nSamples] = size(eegData);
    t = (0:nSamples-1)/fs;
    
    % Plot first 8 channels or all if less
    nPlot = min(8, nChannels);
    
    hold on;
    colors = lines(nPlot);
    
    for ch = 1:nPlot
        % Normalize each channel for better visualization
        signal = eegData(ch, :);
        signal = signal / max(abs(signal)) * 0.8;
        
        plot(t, signal + ch, 'Color', colors(ch, :), 'LineWidth', 0.5);
    end
    
    % Add event markers
    eventIdx = find(triggers > 0);
    if ~isempty(eventIdx)
        eventTimes = eventIdx/fs;
        eventClasses = triggers(eventIdx);
        
        % Use different colors for different classes
        classColors = lines(max(eventClasses));
        
        for i = 1:length(eventTimes)
            class = eventClasses(i);
            plot([eventTimes(i), eventTimes(i)], [0.5, nPlot+0.5], ...
                 'Color', classColors(class, :), 'LineWidth', 1, 'LineStyle', '--');
        end
    end
    
    ylim([0.5, nPlot+0.5]);
    xlabel('Time (s)');
    ylabel('Channel');
    set(gca, 'YTick', 1:nPlot, 'YTickLabel', arrayfun(@(x) sprintf('Ch%d', x), 1:nPlot, 'UniformOutput', false));
    grid on;
    hold off;
    
    % Set colors
    set(gca, 'Color', [0.1, 0.1, 0.15]);
    set(gca, 'XColor', 'white');
    set(gca, 'YColor', 'white');
    set(gca, 'GridColor', [0.3, 0.3, 0.3]);
end

function plotPowerSpectrum(eegData, fs)
    % Plot average power spectrum
    
    [nChannels, nSamples] = size(eegData);
    
    % Compute average PSD
    avgPSD = zeros(1, 512);
    nUsed = 0;
    
    for ch = 1:min(8, nChannels) % Use first 8 channels
        [pxx, f] = pwelch(eegData(ch, :), [], [], [], fs);
        
        % Interpolate to standard frequency vector
        fInterp = linspace(0, fs/2, 512);
        pxxInterp = interp1(f, pxx, fInterp, 'linear', 0);
        
        avgPSD = avgPSD + pxxInterp;
        nUsed = nUsed + 1;
    end
    
    avgPSD = avgPSD / nUsed;
    
    % Plot
    plot(fInterp, 10*log10(avgPSD + eps), 'b-', 'LineWidth', 2);
    
    % Add frequency band annotations
    hold on;
    
    bands = struct('name', {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'}, ...
                   'range', {[1,4], [4,8], [8,13], [13,30], [30,45]}, ...
                   'color', {[0, 0.5, 1], [0.5, 0, 1], [1, 0.5, 0], [0, 1, 0.5], [1, 0, 0.5]});
    
    for b = 1:length(bands)
        idx = fInterp >= bands(b).range(1) & fInterp <= bands(b).range(2);
        if any(idx)
            area(fInterp(idx), 10*log10(avgPSD(idx) + eps), ...
                 'FaceColor', bands(b).color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
    end
    
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    grid on;
    xlim([0, 50]);
    hold off;
    
    % Set colors
    set(gca, 'Color', [0.1, 0.1, 0.15]);
    set(gca, 'XColor', 'white');
    set(gca, 'YColor', 'white');
    set(gca, 'GridColor', [0.3, 0.3, 0.3]);
end

function plotTopographicalMap(eegData)
    % Create simple topographical map
    
    [nChannels, ~] = size(eegData);
    
    % Compute average power for each channel
    channelPower = var(eegData, [], 2);
    
    % Create circular layout
    theta = linspace(0, 2*pi, nChannels+1);
    theta = theta(1:end-1);
    x = cos(theta);
    y = sin(theta);
    
    % Create grid for interpolation
    [xi, yi] = meshgrid(linspace(-1.2, 1.2, 50));
    
    % Interpolate using scattered data
    F = scatteredInterpolant(x', y', channelPower, 'natural', 'nearest');
    zi = F(xi, yi);
    
    % Create mask for circular shape
    mask = sqrt(xi.^2 + yi.^2) <= 1;
    zi(~mask) = NaN;
    
    % Plot
    contourf(xi, yi, zi, 20, 'LineStyle', 'none');
    hold on;
    
    % Plot channel locations
    scatter(x, y, 100, channelPower, 'filled', 'MarkerEdgeColor', 'white');
    
    % Add nose and ears for orientation
    plot([-0.15, 0, 0.15], [1.1, 1.2, 1.1], 'k-', 'LineWidth', 2); % Nose
    plot([-1.2, -1.3], [0, 0], 'k-', 'LineWidth', 2); % Left ear
    plot([1.2, 1.3], [0, 0], 'k-', 'LineWidth', 2);   % Right ear
    
    colormap(jet);
    colorbar;
    axis equal;
    axis off;
    hold off;
    
    set(gca, 'Color', [0.1, 0.1, 0.15]);
end

function plotModelPerformance(accuracy)
    % Plot model performance comparison
    
    % Extract model names and accuracies
    modelNames = fieldnames(accuracy);
    accValues = [];
    validModels = {};
    
    for i = 1:length(modelNames)
        if ~strcmp(modelNames{i}, 'confusionMatrix')
            accValues(end+1) = accuracy.(modelNames{i}) * 100;
            validModels{end+1} = modelNames{i};
        end
    end
    
    if isempty(accValues)
        text(0.5, 0.5, 'No model data', 'HorizontalAlignment', 'center', 'Color', 'white');
        axis off;
        return;
    end
    
    % Create bar plot
    h = bar(accValues);
    
    % Color bars based on performance
    colors = zeros(length(accValues), 3);
    for i = 1:length(accValues)
        if accValues(i) >= 90
            colors(i, :) = [0, 0.8, 0]; % Green for excellent
        elseif accValues(i) >= 80
            colors(i, :) = [1, 0.8, 0]; % Yellow for good
        elseif accValues(i) >= 70
            colors(i, :) = [1, 0.5, 0]; % Orange for fair
        else
            colors(i, :) = [1, 0, 0]; % Red for poor
        end
    end
    
    h.FaceColor = 'flat';
    h.CData = colors;
    
    % Add value labels
    for i = 1:length(accValues)
        text(i, accValues(i) + 1, sprintf('%.1f%%', accValues(i)), ...
             'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
    
    % Format plot
    set(gca, 'XTickLabel', validModels);
    ylabel('Accuracy (%)');
    ylim([0, 105]);
    grid on;
    
    % Highlight best model
    [bestAcc, bestIdx] = max(accValues);
    text(bestIdx, bestAcc/2, 'BEST', ...
         'HorizontalAlignment', 'center', 'Color', 'white', ...
         'FontWeight', 'bold', 'FontSize', 12);
    
    % Set colors
    set(gca, 'Color', [0.1, 0.1, 0.15]);
    set(gca, 'XColor', 'white');
    set(gca, 'YColor', 'white');
    set(gca, 'GridColor', [0.3, 0.3, 0.3]);
end

function plotFeatureImportance(features, models)
    % Plot feature importance from Random Forest
    
    if ~isfield(models, 'rf') || isempty(models.rf)
        text(0.5, 0.5, 'No Random Forest model available', ...
             'HorizontalAlignment', 'center', 'Color', 'white');
        axis off;
        return;
    end
    
    try
        % Get feature importance
        imp = models.rf.OOBPermutedPredictorDeltaError;
        
        if isempty(imp)
            text(0.5, 0.5, 'No feature importance data', ...
                 'HorizontalAlignment', 'center', 'Color', 'white');
            axis off;
            return;
        end
        
        % Sort features by importance
        [sortedImp, idx] = sort(imp, 'descend');
        
        % Plot top 20 features
        nTop = min(20, length(sortedImp));
        barh(sortedImp(1:nTop));
        
        % Add feature names if available
        if nTop > 0
            set(gca, 'YTick', 1:nTop);
            
            % Create simple feature labels
            yLabels = arrayfun(@(x) sprintf('F%03d', idx(x)), 1:nTop, 'UniformOutput', false);
            set(gca, 'YTickLabel', yLabels);
        end
        
        xlabel('Importance (OOB Error Increase)');
        title('Top 20 Important Features', 'Color', 'white');
        grid on;
        
    catch
        text(0.5, 0.5, 'Error plotting feature importance', ...
             'HorizontalAlignment', 'center', 'Color', 'white');
    end
    
    % Set colors
    set(gca, 'Color', [0.1, 0.1, 0.15]);
    set(gca, 'XColor', 'white');
    set(gca, 'YColor', 'white');
    set(gca, 'GridColor', [0.3, 0.3, 0.3]);
end

function plotTimeFrequency(signal, fs)
    % Plot time-frequency representation using spectrogram
    
    % Compute spectrogram
    [s, f, t] = spectrogram(signal, 256, 250, 256, fs);
    
    % Convert to dB
    sdB = 10*log10(abs(s) + eps);
    
    % Plot
    imagesc(t, f, sdB);
    axis xy;
    colormap(jet);
    colorbar;
    
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    ylim([0, 50]);
    
    % Set colors
    set(gca, 'Color', [0.1, 0.1, 0.15]);
    set(gca, 'XColor', 'white');
    set(gca, 'YColor', 'white');
end

function plotConfusionMatrix(confMat, classNames)
    % Plot confusion matrix
    
    if nargin < 2
        classNames = arrayfun(@(x) sprintf('Class %d', x), 1:size(confMat, 1), 'UniformOutput', false);
    end
    
    % Normalize by row
    confMatNorm = confMat ./ sum(confMat, 2);
    confMatNorm(isnan(confMatNorm)) = 0;
    
    % Create heatmap
    imagesc(confMatNorm);
    
    % Add text
    for i = 1:size(confMat, 1)
        for j = 1:size(confMat, 2)
            text(j, i, sprintf('%d\n(%.1f%%)', confMat(i, j), confMatNorm(i, j)*100), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Color', 'white', 'FontWeight', 'bold');
        end
    end
    
    % Format
    colormap(flipud(gray));
    colorbar;
    
    set(gca, 'XTick', 1:size(confMat, 2), 'XTickLabel', classNames);
    set(gca, 'YTick', 1:size(confMat, 1), 'YTickLabel', classNames);
    xlabel('Predicted Class');
    ylabel('True Class');
    
    % Set colors
    set(gca, 'Color', [0.1, 0.1, 0.15]);
    set(gca, 'XColor', 'white');
    set(gca, 'YColor', 'white');
end

function plotSystemInfo(eegData, features, models, accuracy, config)
    % Display system information
    
    infoText = {
        sprintf('NEURONSYNC SYSTEM INFO');
        sprintf('======================');
        sprintf('Data Information:');
        sprintf('  Channels: %d', size(eegData, 1));
        sprintf('  Samples: %d', size(eegData, 2));
        sprintf('  Duration: %.1f s', size(eegData, 2)/config.sampling_rate);
        sprintf('  Sampling Rate: %d Hz', config.sampling_rate);
        sprintf('');
        sprintf('Feature Information:');
        sprintf('  Features Extracted: %d', size(features, 2));
        sprintf('  Feature Windows: %d', size(features, 1));
        sprintf('');
        sprintf('Model Information:');
    };
    
    % Add model accuracies
    modelNames = fieldnames(accuracy);
    for i = 1:length(modelNames)
        if ~strcmp(modelNames{i}, 'confusionMatrix')
            infoText{end+1} = sprintf('  %s: %.1f%%', ...
                modelNames{i}, accuracy.(modelNames{i})*100);
        end
    end
    
    infoText{end+1} = '';
    infoText{end+1} = sprintf('Best Model: %s', ...
        accuracy.bestModel);
    
    % Display as text
    text(0.1, 0.5, strjoin(infoText, '\n'), ...
         'FontName', 'Monospaced', 'FontSize', 9, ...
         'VerticalAlignment', 'middle', 'Color', 'white');
    
    axis off;
    
    % Set background
    set(gca, 'Color', [0.1, 0.1, 0.15]);
end