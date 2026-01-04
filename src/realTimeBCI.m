function realTimeBCI(models, config)
    % REALTIMEBCI - Real-time Brain-Computer Interface interface
    %
    % Inputs:
    %   models - Trained ML models
    %   config - Configuration structure
    
    fprintf('   Launching real-time BCI interface...\n');
    
    % Create figure
    fig = figure('Name', 'NeuroSync - Real-time BCI', ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 1200, 700], ...
                 'Color', [0.05, 0.05, 0.1], ...
                 'CloseRequestFcn', @closeFig);
    
    % Initialize data
    fs = config.sampling_rate;
    bufferSize = 5 * fs;  % 5-second buffer
    eegBuffer = zeros(config.channels, bufferSize);
    predictionHistory = [];
    timeVector = (0:bufferSize-1)/fs;
    
    % Create subplots
    ax1 = subplot(2, 3, [1, 2]);
    ax2 = subplot(2, 3, 3);
    ax3 = subplot(2, 3, [4, 6]);
    
    % UI Controls
    uicontrol('Style', 'text', 'Position', [10, 650, 300, 30], ...
              'String', 'NEURONSYNC - Real-time BCI', ...
              'FontSize', 14, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.05, 0.05, 0.1], ...
              'ForegroundColor', 'white');
    
    startBtn = uicontrol('Style', 'pushbutton', ...
                        'Position', [10, 600, 100, 40], ...
                        'String', 'START', ...
                        'FontSize', 11, 'FontWeight', 'bold', ...
                        'BackgroundColor', [0, 0.6, 0], ...
                        'ForegroundColor', 'white', ...
                        'Callback', @startCallback);
    
    stopBtn = uicontrol('Style', 'pushbutton', ...
                       'Position', [120, 600, 100, 40], ...
                       'String', 'STOP', ...
                       'FontSize', 11, 'FontWeight', 'bold', ...
                       'BackgroundColor', [0.6, 0, 0], ...
                       'ForegroundColor', 'white', ...
                       'Callback', @stopCallback);
    
    statusText = uicontrol('Style', 'text', ...
                          'Position', [10, 550, 200, 40], ...
                          'String', 'Status: Ready', ...
                          'FontSize', 10, ...
                          'BackgroundColor', [0.05, 0.05, 0.1], ...
                          'ForegroundColor', 'yellow');
    
    % Control variables
    isRunning = false;
    timerObj = [];
    
    % Callback functions
    function startCallback(~, ~)
        if ~isRunning
            isRunning = true;
            set(statusText, 'String', 'Status: Running', 'ForegroundColor', 'green');
            
            % Start timer for updates
            timerObj = timer('ExecutionMode', 'fixedRate', ...
                            'Period', 0.1, ...
                            'TimerFcn', @updateDisplay);
            start(timerObj);
        end
    end
    
    function stopCallback(~, ~)
        if isRunning
            isRunning = false;
            set(statusText, 'String', 'Status: Stopped', 'ForegroundColor', 'red');
            
            if ~isempty(timerObj)
                stop(timerObj);
                delete(timerObj);
            end
        end
    end
    
    function closeFig(~, ~)
        % Clean up on close
        if ~isempty(timerObj)
            stop(timerObj);
            delete(timerObj);
        end
        delete(fig);
    end
    
    function updateDisplay(~, ~)
        if ~isRunning || ~ishandle(fig)
            return;
        end
        
        % Generate simulated EEG data for demonstration
        simulatedData = generateRealTimeEEG(config.channels, fs, 0.1);
        
        % Update buffer (FIFO)
        eegBuffer = [eegBuffer(:, size(simulatedData, 2)+1:end), simulatedData];
        
        % Extract features from last 2 seconds
        featureWindow = 2 * fs;
        if size(eegBuffer, 2) >= featureWindow
            windowData = eegBuffer(:, end-featureWindow+1:end);
            features = extractEEGFeatures(windowData, config);
            
            if ~isempty(features) && ~isempty(models)
                % Make prediction using best model
                if isfield(models, 'svm') && ~isempty(models.svm)
                    prediction = predict(models.svm, features(end, :));
                    
                    % Update prediction history
                    predictionHistory = [predictionHistory, prediction];
                    if length(predictionHistory) > 20
                        predictionHistory = predictionHistory(end-19:end);
                    end
                end
            end
        end
        
        % Update all displays
        updateEEGPlot(ax1, eegBuffer, timeVector);
        updateSpectrumPlot(ax2, eegBuffer, fs);
        updatePredictionPlot(ax3, predictionHistory, config);
        
        drawnow;
    end
    
    % Initial display
    updateDisplay([], []);
    
    fprintf('   Real-time BCI ready. Press START to begin.\n');
end

function data = generateRealTimeEEG(nChannels, fs, duration)
    % Generate simulated real-time EEG data
    
    nSamples = round(fs * duration);
    data = zeros(nChannels, nSamples);
    t = (0:nSamples-1)/fs;
    
    % Add base rhythms with random phase
    for ch = 1:nChannels
        alpha = 0.5 * sin(2*pi*10*t + rand()*2*pi);
        beta = 0.3 * sin(2*pi*20*t + rand()*2*pi);
        noise = 0.1 * randn(1, nSamples);
        
        % Channel-specific variation
        gain = 0.8 + 0.4*(ch/nChannels);
        data(ch, :) = gain * (alpha + beta + noise);
    end
end

function updateEEGPlot(ax, eegBuffer, timeVector)
    % Update EEG time series plot
    
    cla(ax);
    
    % Plot first few channels
    nPlot = min(6, size(eegBuffer, 1));
    
    hold(ax, 'on');
    colors = lines(nPlot);
    
    for ch = 1:nPlot
        signal = eegBuffer(ch, :);
        signal = signal / max(abs(signal)) * 0.8; % Normalize
        
        plot(ax, timeVector, signal + ch, ...
             'Color', colors(ch, :), 'LineWidth', 1);
    end
    
    hold(ax, 'off');
    
    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Channel');
    ylim(ax, [0.5, nPlot+0.5]);
    set(ax, 'YTick', 1:nPlot);
    set(ax, 'YTickLabel', arrayfun(@(x) sprintf('Ch%d', x), 1:nPlot, 'UniformOutput', false));
    title(ax, 'Real-time EEG', 'Color', 'white');
    grid(ax, 'on');
    
    % Set colors
    set(ax, 'Color', [0.1, 0.1, 0.15]);
    set(ax, 'XColor', 'white');
    set(ax, 'YColor', 'white');
    set(ax, 'GridColor', [0.3, 0.3, 0.3]);
end

function updateSpectrumPlot(ax, eegBuffer, fs)
    % Update power spectrum plot
    
    cla(ax);
    
    % Compute PSD for first channel
    if ~isempty(eegBuffer)
        [pxx, f] = pwelch(eegBuffer(1, :), [], [], [], fs);
        
        plot(ax, f, 10*log10(pxx + eps), 'b-', 'LineWidth', 2);
        
        xlabel(ax, 'Frequency (Hz)');
        ylabel(ax, 'Power (dB)');
        title(ax, 'Power Spectrum (Ch1)', 'Color', 'white');
        grid(ax, 'on');
        xlim(ax, [0, 50]);
    end
    
    % Set colors
    set(ax, 'Color', [0.1, 0.1, 0.15]);
    set(ax, 'XColor', 'white');
    set(ax, 'YColor', 'white');
    set(ax, 'GridColor', [0.3, 0.3, 0.3]);
end

function updatePredictionPlot(ax, predictionHistory, config)
    % Update prediction display
    
    cla(ax);
    
    if isempty(predictionHistory)
        text(ax, 0.5, 0.5, 'No predictions yet', ...
             'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 14);
        axis(ax, 'off');
        return;
    end
    
    % Get current prediction
    currentPred = predictionHistory(end);
    
    % Define colors and labels
    colors = {[1, 0.2, 0.2], [0.2, 0.6, 1], [0.2, 1, 0.2], [1, 0.8, 0.2]};
    labels = {'Left Hand', 'Right Hand', 'Feet', 'Tongue'};
    
    if currentPred > 0 && currentPred <= length(labels)
        % Draw brain schematic
        drawBrainSchematic(ax, currentPred, colors);
        
        % Display current prediction
        text(ax, 0.5, 0.9, sprintf('DETECTED: %s', labels{currentPred}), ...
             'HorizontalAlignment', 'center', 'FontSize', 24, 'FontWeight', 'bold', ...
             'Color', colors{currentPred});
        
        % Display confidence
        if length(predictionHistory) > 5
            confidence = sum(predictionHistory == currentPred) / length(predictionHistory);
            text(ax, 0.5, 0.15, sprintf('Confidence: %.1f%%', confidence * 100), ...
                 'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', 'white');
        end
    end
    
    axis(ax, 'off');
    set(ax, 'Color', [0.05, 0.05, 0.1]);
end

function drawBrainSchematic(ax, activeRegion, colors)
    % Draw simple brain schematic with active region highlighted
    
    % Brain outline
    theta = linspace(0, 2*pi, 100);
    x_outline = cos(theta) * 0.4 + 0.5;
    y_outline = sin(theta) * 0.3 + 0.5;
    
    plot(ax, x_outline, y_outline, 'w-', 'LineWidth', 2);
    hold(ax, 'on');
    
    % Region positions
    regions = {
        [0.3, 0.5],  % Left hand (right hemisphere)
        [0.7, 0.5],  % Right hand (left hemisphere)
        [0.5, 0.3],  % Feet
        [0.5, 0.7]   % Tongue
    };
    
    % Draw all regions
    for r = 1:length(regions)
        if r == activeRegion
            % Active region - larger and colored
            plot(ax, regions{r}(1), regions{r}(2), 'o', ...
                 'MarkerSize', 40, 'LineWidth', 3, ...
                 'MarkerFaceColor', colors{r}, 'Color', 'white');
        else
            % Inactive regions - smaller and gray
            plot(ax, regions{r}(1), regions{r}(2), 'o', ...
                 'MarkerSize', 20, 'LineWidth', 1, ...
                 'MarkerFaceColor', [0.3, 0.3, 0.3], 'Color', [0.5, 0.5, 0.5]);
        end
    end
    
    % Add labels
    text(ax, 0.3, 0.55, 'Left', 'HorizontalAlignment', 'center', 'Color', 'white');
    text(ax, 0.7, 0.55, 'Right', 'HorizontalAlignment', 'center', 'Color', 'white');
    text(ax, 0.5, 0.25, 'Feet', 'HorizontalAlignment', 'center', 'Color', 'white');
    text(ax, 0.5, 0.75, 'Tongue', 'HorizontalAlignment', 'center', 'Color', 'white');
    
    hold(ax, 'off');
end