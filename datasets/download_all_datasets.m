function download_all_datasets()
    % DOWNLOAD_ALL_DATASETS - Download all datasets for NeuroSync
    
    fprintf('====================================\n');
    fprintf('📥 NeuroSync Dataset Downloader\n');
    fprintf('====================================\n\n');
    
    fprintf('This script helps you download datasets for NeuroSync.\n');
    fprintf('Due to size constraints, we provide sample data by default.\n\n');
    
    fprintf('Available options:\n');
    fprintf('1. Use built-in sample data (already included)\n');
    fprintf('2. Download BCI Competition IV Dataset 2a (2.5 GB)\n');
    fprintf('3. Download PhysioNet EEG MMIDB (1.5 GB)\n');
    fprintf('4. Generate enhanced sample data\n\n');
    
    choice = input('Enter your choice (1-4): ', 's');
    
    switch choice
        case '1'
            fprintf('\nUsing built-in sample data...\n');
            fprintf('Sample data is already included in datasets/sample_data.mat\n');
            
        case '2'
            fprintf('\nBCI Competition IV Dataset 2a:\n');
            fprintf('Official URL: http://bnci-horizon-2020.eu/database/data-sets/001-2014\n');
            fprintf('Please download manually and extract to datasets/bci_iv_2a/\n');
            
        case '3'
            fprintf('\nPhysioNet EEG MMIDB:\n');
            fprintf('URL: https://physionet.org/content/eegmmidb/1.0.0/\n');
            fprintf('Use wget command: wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/\n');
            
        case '4'
            fprintf('\nGenerating enhanced sample data...\n');
            generate_enhanced_sample();
            
        otherwise
            fprintf('\nInvalid choice. Using built-in sample data.\n');
    end
    
    fprintf('\n====================================\n');
    fprintf('Dataset setup complete!\n');
    fprintf('Run NeuroSync() to start the system.\n');
    fprintf('====================================\n');
end

function generate_enhanced_sample()
    % Generate enhanced sample data
    
    fprintf('Creating enhanced sample data...\n');
    
    % Parameters
    fs = 250;
    duration = 300; % 5 minutes
    nChannels = 22;
    nSamples = fs * duration;
    
    % Create directory if needed
    if ~exist('datasets', 'dir')
        mkdir('datasets');
    end
    
    % Generate EEG data with realistic characteristics
    t = (0:nSamples-1)/fs;
    
    % Initialize data
    eegData = zeros(nChannels, nSamples);
    
    % Channel-specific characteristics
    channelTypes = {
        'frontal', [1, 2, 3, 4];
        'central', [5, 6, 7, 8, 9, 10, 11];
        'parietal', [12, 13, 14, 15, 16];
        'occipital', [17, 18];
        'temporal', [19, 20, 21, 22]
    };
    
    for typeIdx = 1:size(channelTypes, 1)
        typeName = channelTypes{typeIdx, 1};
        channels = channelTypes{typeIdx, 2};
        
        for ch = channels
            % Base noise
            signal = 0.1 * randn(1, nSamples);
            
            % Add frequency characteristics based on region
            switch typeName
                case 'frontal'
                    % More theta and beta
                    theta = 0.4 * sin(2*pi*6*t + rand()*2*pi);
                    beta = 0.3 * sin(2*pi*20*t + rand()*2*pi);
                    signal = signal + theta + beta;
                    
                case 'central'
                    % More mu rhythm (8-13 Hz) for motor areas
                    mu = 0.6 * sin(2*pi*10*t + rand()*2*pi);
                    signal = signal + mu;
                    
                case 'parietal'
                    % More alpha
                    alpha = 0.5 * sin(2*pi*10*t + rand()*2*pi);
                    signal = signal + alpha;
                    
                case 'occipital'
                    % Strong alpha
                    alpha = 0.8 * sin(2*pi*10*t + rand()*2*pi);
                    signal = signal + alpha;
                    
                case 'temporal'
                    % Mixed frequencies
                    theta = 0.3 * sin(2*pi*6*t + rand()*2*pi);
                    alpha = 0.3 * sin(2*pi*10*t + rand()*2*pi);
                    signal = signal + theta + alpha;
            end
            
            eegData(ch, :) = signal;
        end
    end
    
    % Create events
    triggers = create_complex_events(fs, nSamples);
    
    % Add ERD/ERS effects
    eegData = add_complex_effects(eegData, triggers, fs);
    
    % Create metadata
    metadata = struct();
    metadata.creation_date = datestr(now);
    metadata.sampling_rate = fs;
    metadata.duration = duration;
    metadata.n_channels = nChannels;
    metadata.channel_labels = {'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',...
                              'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Oz',...
                              'FC1','FC2'};
    metadata.class_names = {'Left Hand', 'Right Hand', 'Feet', 'Tongue'};
    metadata.description = 'Enhanced sample EEG data for NeuroSync';
    
    % Save
    save('datasets/enhanced_sample.mat', 'eegData', 'triggers', 'metadata', '-v7.3');
    
    fprintf('Enhanced sample data saved to: datasets/enhanced_sample.mat\n');
    fprintf('Size: %d channels x %d samples (%.1f MB)\n', ...
        size(eegData, 1), size(eegData, 2), ...
        getfield(dir('datasets/enhanced_sample.mat'), 'bytes')/1024/1024);
end

function triggers = create_complex_events(fs, nSamples)
    % Create complex event structure
    
    triggers = zeros(1, nSamples);
    
    % Event types and durations
    events = [
        5, 4, 1;   % Start at 5s, duration 4s, class 1
        15, 4, 2;  % Start at 15s, duration 4s, class 2
        25, 4, 3;  % Start at 25s, duration 4s, class 3
        35, 4, 4;  % Start at 35s, duration 4s, class 4
        50, 6, 1;  % Longer event
        65, 3, 2;  % Shorter event
        80, 4, 3;
        95, 4, 4;
        110, 5, 1;
        125, 4, 2;
        140, 4, 3;
        155, 4, 4;
        170, 6, 1;
        185, 3, 2;
        200, 4, 3;
        215, 4, 4;
        230, 5, 1;
        245, 4, 2;
        260, 4, 3;
        275, 4, 4
    ];
    
    for i = 1:size(events, 1)
        startIdx = round(events(i, 1) * fs);
        durationIdx = round(events(i, 2) * fs);
        classId = events(i, 3);
        
        endIdx = min(startIdx + durationIdx - 1, nSamples);
        if startIdx <= nSamples
            triggers(startIdx:endIdx) = classId;
        end
    end
end

function eegData = add_complex_effects(eegData, triggers, fs)
    % Add complex ERD/ERS effects
    
    eventIdx = find(triggers > 0);
    
    if isempty(eventIdx)
        return;
    end
    
    % Group events
    eventGroups = {};
    currentGroup = [];
    
    for i = 1:length(eventIdx)
        if isempty(currentGroup)
            currentGroup = eventIdx(i);
        elseif eventIdx(i) == currentGroup(end) + 1
            currentGroup(end+1) = eventIdx(i);
        else
            eventGroups{end+1} = currentGroup;
            currentGroup = eventIdx(i);
        end
    end
    
    if ~isempty(currentGroup)
        eventGroups{end+1} = currentGroup;
    end
    
    % Apply effects to each event
    for g = 1:length(eventGroups)
        indices = eventGroups{g};
        if length(indices) < fs/2
            continue;
        end
        
        classId = triggers(indices(1));
        
        % Determine affected channels and effect strength
        switch classId
            case 1  % Left hand
                affectedChannels = [5, 6, 7];  % C3, C4, Cz region
                erdStrength = 0.6 + 0.2*rand(); % 40-20% reduction
                effectType = 'ERD'; % Event-Related Desynchronization
                
            case 2  % Right hand
                affectedChannels = [8, 9, 10]; % More frontal
                erdStrength = 0.65 + 0.15*rand();
                effectType = 'ERD';
                
            case 3  % Feet
                affectedChannels = [3, 4, 11, 12]; % Central-parietal
                ersStrength = 1.2 + 0.3*rand(); % 20-50% increase (ERS)
                effectType = 'ERS';
                
            case 4  % Tongue
                affectedChannels = [1, 2, 13, 14]; % Frontal-temporal
                erdStrength = 0.7 + 0.1*rand();
                effectType = 'ERD';
                
            otherwise
                affectedChannels = 1:min(4, size(eegData, 1));
                erdStrength = 0.8;
                effectType = 'ERD';
        end
        
        % Apply effect with tapered window
        nPoints = length(indices);
        window = tukeywin(nPoints, 0.4)';
        
        for ch = affectedChannels
            if ch <= size(eegData, 1)
                switch effectType
                    case 'ERD'
                        % Power decrease
                        effectPattern = 1 - (1-erdStrength) * window;
                    case 'ERS'
                        % Power increase
                        effectPattern = ersStrength * window + (1 - window);
                end
                
                eegData(ch, indices) = eegData(ch, indices) .* effectPattern;
            end
        end
    end
end