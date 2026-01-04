function [eegData, triggers, channelInfo, config] = loadEEGData(datasetName, options)
    % LOADEEGDATA - Load EEG dataset from various sources
    %
    % Inputs:
    %   datasetName - 'simulated', 'bci_iv_2a', 'physionet', 'deap', 'sample'
    %   options - Structure with loading options
    %
    % Outputs:
    %   eegData - EEG data [channels x samples]
    %   triggers - Event markers
    %   channelInfo - Channel information
    %   config - Dataset configuration

    if nargin < 2
        options = struct();
    end
    
    fprintf('   Loading dataset: %s\n', datasetName);

    % Dataset Configurations
    switch lower(datasetName)
        case 'simulated'
            config = struct( ...
                'name', 'Simulated EEG Data',...
                'sampling_rate', 250,...
                'channels', 22,...
                'classes', {'Left Hand', 'Right Hand', 'Feet', 'Tongue'},...
                'class_ids', [1, 2, 3, 4]...
                );

            % Generate Simulated Data
            [eegData, triggers] = generateSimulatedEEG(config, options);
            channelInfo = createChannelInfo(config.channels);

        case 'sample'
            config = struct( ...
                'name', 'Sample EEG Data',...
                'sampling_rate', 250,...
                'channels', 22,...
                'classes', {'Class 1', 'Class 2', 'Class 3', 'Class 4'},...
                'class_ids', [1, 2, 3, 4]...
                );

            % Load Sample Data
            [eegData, triggers, channelInfo] = loadSampleData();

        case 'bci_iv_2a'
            config = struct( ...
                'name', 'BCI Competition IV Dataset 2a',...
                'sampling_rate', 250,...
                'channels', 22,...
                'classes', {'Left Hand', 'Right Hand', 'Feet', 'Tongue'},...
                'class_ids', [1, 2, 3, 4]...
                );
        
        % Try to load BCI data
        [eegData, triggers, channelInfo] = loadBCIData(options);

        otherwise
            error('Dataset %s not supported. Use: simulated, sample, or bci_iv_2a', datasetName);
    end

    fprintf('   Loaded: %d channels, %d samples\n', size(eegData, 1), size(eegData, 2));
end

function [eegData, triggers] = generateSimulatedEEG(config, options)
    % Generate realistic simulated EEG Data

    % default Parameters
    if ~isfield(options, 'duration')
        options.duration = 300; % 5 Minutes
    end
    if ~isfield(options, 'subject')
        options.subject = 1;
    end

    fs = config.sampling_rate;
    nChannels = config.channels;
    nSamples = fs * options.duration;
    t = (0:nSamples-1)/fs;

    fprintf('   Generating simulated data for subject %d...\n', options.subject);

    % Initialize data matrix
    eegData = zeros(nChannels, nSamples);

    % Generate background EEG for each channel
    for ch = 1:nChannels
        % Alpha rythm (8-12 Hz) - relaxed state
        alphaFreq = 10 + 2*randn();
        alphaAmp = 0.5 + 0.2*randn();
        alpha = alphaAmp * sin(2*pi*alphaFreq*t + rand()*2*pi);

        % Beat rythm (13-30 Hz) - active thinking
        betaFreq = 20 + 5*randn();
        betaAmp = 0.3 * 0.1*randn();
        beta = betaAmp * sin(2*pi*betaFreq*t + rand()*2*pi);

        % Theta rythm (4-7 Hz) - drowsiness
        thetaFreq = 6 + 1*randn();
        thetaAmp = 0.4 + 0.1*randn();
        theta = thetaAmp * sin(2*pi*thetaFreq*t + rand()*2*pi);

        % Combine with Gaussian Noise
        noise = 0.1 * randn(1, nSamples);

        % Channel-specific variation
        channelGain = 0.8 + 0.4*(ch/nChannels);
        eegData(ch, :) = channelGain * (alpha + beta + theta + noise);
    end

    % Create event markers (motor imagenary tasks)
    triggers = createSimulatedEvents(fs, nSamples, config);

    % Add Event-Related Desynchronizaton (ERD) effects
    eegData = addERDEffects(eegData, triggers, fs);
end

function triggers = createSimulatedEvents(fs, nSamples, config)
    % Create simulated event markers
    
    triggers = zeros(1, nSamples);
    
    % Event parameters
    eventDuration = 4 * fs;      % 4 seconds per event
    interEventInterval = 8 * fs; % 8 seconds between events
    startTime = 5 * fs;          % Start 5 seconds in
    
    % Create 20 events
    nEvents = 20;
    
    for ev = 1:nEvents
        startIdx = startTime + (ev-1)*(eventDuration + interEventInterval);
        endIdx = min(startIdx + eventDuration - 1, nSamples);
        
        if endIdx > nSamples
            break;
        end
        
        % Assign class (1-4) in round-robin fashion
        classId = mod(ev-1, 4) + 1;
        triggers(startIdx:endIdx) = classId;
    end
end

function eegData = addERDEffects(eegData, triggers, fs)
    % Add Event-Related Desynchronization effects
    
    nChannels = size(eegData, 1);
    eventIndices = find(triggers > 0);
    
    if isempty(eventIndices)
        return;
    end
    
    % Group consecutive indices
    eventGroups = {};
    currentGroup = [];
    
    for i = 1:length(eventIndices)
        if isempty(currentGroup)
            currentGroup = eventIndices(i);
        elseif eventIndices(i) == currentGroup(end) + 1
            currentGroup(end+1) = eventIndices(i);
        else
            eventGroups{end+1} = currentGroup;
            currentGroup = eventIndices(i);
        end
    end
    
    if ~isempty(currentGroup)
        eventGroups{end+1} = currentGroup;
    end
    
    % Apply ERD to each event
    for g = 1:length(eventGroups)
        indices = eventGroups{g};
        if length(indices) < fs  % Skip very short events
            continue;
        end
        
        classId = triggers(indices(1));
        
        % Determine affected channels based on class
        switch classId
            case 1  % Left hand - right hemisphere
                affectedChannels = [7, 8, 9, 15, 16]; % C3, CP3 regions
                erdStrength = 0.7; % 30% reduction
                
            case 2  % Right hand - left hemisphere
                affectedChannels = [11, 12, 13, 17, 18]; % C4, CP4 regions
                erdStrength = 0.7;
                
            case 3  % Feet - central region
                affectedChannels = [9, 10, 11, 16, 17]; % Cz, CPz regions
                erdStrength = 0.8; % 20% reduction
                
            case 4  % Tongue - frontal-central
                affectedChannels = [2, 3, 4, 5, 6]; % FC regions
                erdStrength = 0.75;
                
            otherwise
                affectedChannels = 1:min(5, nChannels);
                erdStrength = 0.8;
        end
        
        % Apply ERD effect
        for ch = affectedChannels
            if ch <= nChannels
                % Gradual decrease and recovery
                nPoints = length(indices);
                window = tukeywin(nPoints, 0.5)'; % Tapered window
                erdPattern = 1 - (1-erdStrength) * window;
                
                eegData(ch, indices) = eegData(ch, indices) .* erdPattern;
            end
        end
    end
end

function [eegData, triggers, channelInfo] = loadSampleData()
    % Load sample data from file
    
    sampleFile = 'datasets/sample_data.mat';
    
    if exist(sampleFile, 'file')
        data = load(sampleFile);
        
        if isfield(data, 'sampleData') && isfield(data.sampleData, 'motor_imagery')
            % Use motor imagery data
            eegData = data.sampleData.motor_imagery.eeg;
            triggers = data.sampleData.motor_imagery.triggers;
            
            % Create channel info
            channelInfo = struct();
            channelInfo.labels = data.sampleData.motor_imagery.channel_labels;
            channelInfo.types = repmat({'EEG'}, size(channelInfo.labels));
            
            fprintf('   Loaded from sample file\n');
        else
            error('Invalid sample file structure');
        end
    else
        % Create sample data if file doesn't exist
        fprintf('   Creating sample data...\n');
        eegData = randn(22, 250*60) * 0.1; % 1 minute of data
        triggers = zeros(1, size(eegData, 2));
        triggers(10*250:14*250-1) = 1; % Event at 10-14 seconds
        triggers(30*250:34*250-1) = 2; % Event at 30-34 seconds
        
        channelInfo = createChannelInfo(22);
        
        % Save for future use
        sampleData.motor_imagery.eeg = eegData;
        sampleData.motor_imagery.triggers = triggers;
        sampleData.motor_imagery.channel_labels = channelInfo.labels;
        
        if ~exist('datasets', 'dir')
            mkdir('datasets');
        end
        save(sampleFile, 'sampleData');
    end
end

function [eegData, triggers, channelInfo] = loadBCIData(options)
    % Try to load BCI Competition data
    
    % Look for data in datasets/bci_iv_2a/
    dataDir = 'datasets/bci_iv_2a';
    
    if exist(dataDir, 'dir')
        % Check for MAT files
        matFiles = dir(fullfile(dataDir, '*.mat'));
        
        if ~isempty(matFiles)
            % Load first available file
            data = load(fullfile(dataDir, matFiles(1).name));
            
            if isfield(data, 'data')
                eegData = data.data';
                if isfield(data, 'classlabel')
                    triggers = data.classlabel';
                else
                    triggers = zeros(1, size(eegData, 2));
                end
                fprintf('   Loaded BCI data from: %s\n', matFiles(1).name);
            else
                error('Invalid BCI data file');
            end
        else
            % No BCI files found, use simulated
            fprintf('   No BCI files found, using simulated data\n');
            config = struct('sampling_rate', 250, 'channels', 22);
            [eegData, triggers] = generateSimulatedEEG(config, options);
        end
    else
        % Directory doesn't exist, use simulated
        fprintf('   BCI directory not found, using simulated data\n');
        config = struct('sampling_rate', 250, 'channels', 22);
        [eegData, triggers] = generateSimulatedEEG(config, options);
    end
    
    channelInfo = createChannelInfo(size(eegData, 1));
end

function channelInfo = createChannelInfo(nChannels)
    % Create channel information structure
    
    channelInfo = struct();
    
    % Standard 10-20 system labels (subset)
    standardLabels = {'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',...
                     'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Oz',...
                     'FC1','FC2','CP1','CP2','FC5','FC6','CP5','CP6'};
    
    if nChannels <= length(standardLabels)
        channelInfo.labels = standardLabels(1:nChannels);
    else
        % Generate generic labels
        channelInfo.labels = arrayfun(@(x) sprintf('Ch%02d', x), 1:nChannels, 'UniformOutput', false);
    end
    
    channelInfo.types = repmat({'EEG'}, size(channelInfo.labels));
    channelInfo.units = repmat({'µV'}, size(channelInfo.labels));
    
    % Add positions if available (simplified)
    theta = linspace(0, 2*pi, nChannels+1);
    theta = theta(1:end-1);
    channelInfo.positions = [cos(theta'); sin(theta')]';
end