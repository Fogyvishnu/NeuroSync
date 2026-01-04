function [eegClean, artifacts, info] = preprocessEEG(eegData, config)
    % PREPROCESSEEG - Complete EEG preprocessing pipeline
    %
    % Inputs:
    %   eegData - Raw EEG data [channels x samples]
    %   config - Configuration structure
    %
    % Outputs:
    %   eegClean - Preprocessed EEG data
    %   artifacts - Artifact information
    %   info - Processing information
    
    [nChannels, nSamples] = size(eegData);
    info = struct();
    info.nChannelsOriginal = nChannels;
    info.nSamplesOriginal = nSamples;
    
    fprintf('   Preprocessing %d channels, %d samples...\n', nChannels, nSamples);
    
    %% 1. Remove DC offset
    eegData = eegData - mean(eegData, 2);
    
    %% 2. Bandpass filter (1-45 Hz)
    fprintf('   Applying bandpass filter (1-45 Hz)...\n');
    fs = config.sampling_rate;
    
    % Design filter
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder', 4, ...
        'HalfPowerFrequency1', 1, ...
        'HalfPowerFrequency2', 45, ...
        'SampleRate', fs, ...
        'DesignMethod', 'butter');
    
    eegFiltered = zeros(size(eegData));
    for ch = 1:nChannels
        eegFiltered(ch, :) = filtfilt(bpFilt, eegData(ch, :));
    end
    
    %% 3. Notch filter (remove powerline noise)
    fprintf('   Applying notch filter (50/60 Hz)...\n');
    
    % Determine notch frequency based on region
    if isfield(config, 'powerline_freq')
        notchFreq = config.powerline_freq;
    else
        notchFreq = 50; % Default to 50 Hz (Europe/Asia)
    end
    
    wo = notchFreq/(fs/2);
    bw = wo/35;
    [b, a] = iirnotch(wo, bw);
    
    for ch = 1:nChannels
        eegFiltered(ch, :) = filtfilt(b, a, eegFiltered(ch, :));
    end
    
    %% 4. Common Average Reference (CAR)
    fprintf('   Applying Common Average Reference...\n');
    avgRef = mean(eegFiltered, 1);
    eegCAR = eegFiltered - avgRef;
    
    %% 5. Artifact Detection
    fprintf('   Detecting artifacts...\n');
    artifacts = detectArtifacts(eegCAR, fs);
    info.artifactPercentage = artifacts.artifactPercentage;
    
    %% 6. Artifact Removal
    fprintf('   Removing artifacts...\n');
    eegClean = removeArtifacts(eegCAR, artifacts);
    
    %% 7. Channel Interpolation (if bad channels detected)
    if sum(artifacts.deadChannels) > 0 && sum(artifacts.deadChannels) < nChannels/2
        fprintf('   Interpolating %d bad channels...\n', sum(artifacts.deadChannels));
        eegClean = interpolateBadChannels(eegClean, artifacts.deadChannels);
    end
    
    %% Update info
    info.nChannelsClean = size(eegClean, 1);
    info.nSamplesClean = size(eegClean, 2);
    
    fprintf('   Preprocessing complete. Artifacts: %.1f%%\n', artifacts.artifactPercentage);
end

function artifacts = detectArtifacts(eegData, fs)
    % Detect various types of artifacts
    
    [nChannels, nSamples] = size(eegData);
    artifacts = struct();
    
    %% 1. High amplitude artifacts (eye blinks, movements)
    amplitudeThreshold = 100; % µV
    highAmpMask = any(abs(eegData) > amplitudeThreshold, 1);
    
    %% 2. Muscle artifacts (high frequency)
    % Calculate RMS in high frequency band
    muscleBand = [30, 100]; % Hz
    [b, a] = butter(4, muscleBand/(fs/2), 'bandpass');
    
    muscleArtifact = false(1, nSamples);
    for ch = 1:min(4, nChannels) % Check frontal channels
        filtered = filtfilt(b, a, eegData(ch, :));
        rms = sqrt(movmean(filtered.^2, fs)); % 1-second window
        muscleArtifact = muscleArtifact | (rms > 3*std(rms));
    end
    
    %% 3. Flatline detection (dead channels)
    channelSTD = std(eegData, [], 2);
    deadThreshold = 0.1; % µV
    deadChannels = channelSTD < deadThreshold;
    
    %% 4. Combined artifact mask
    artifactMask = highAmpMask | muscleArtifact;
    
    %% Store results
    artifacts.highAmpMask = highAmpMask;
    artifacts.muscleArtifact = muscleArtifact;
    artifacts.deadChannels = deadChannels;
    artifacts.artifactMask = artifactMask;
    artifacts.artifactPercentage = sum(artifactMask)/nSamples * 100;
end

function eegClean = removeArtifacts(eegData, artifacts)
    % Remove detected artifacts
    
    [nChannels, nSamples] = size(eegData);
    
    % For now, just zero out artifact segments (could use interpolation)
    eegClean = eegData;
    
    % Create tapered window for artifact segments
    artifactIndices = find(artifacts.artifactMask);
    
    if ~isempty(artifactIndices)
        % Group consecutive artifacts
        artifactGroups = {};
        currentGroup = [];
        
        for i = 1:length(artifactIndices)
            if isempty(currentGroup)
                currentGroup = artifactIndices(i);
            elseif artifactIndices(i) == currentGroup(end) + 1
                currentGroup(end+1) = artifactIndices(i);
            else
                artifactGroups{end+1} = currentGroup;
                currentGroup = artifactIndices(i);
            end
        end
        
        if ~isempty(currentGroup)
            artifactGroups{end+1} = currentGroup;
        end
        
        % Apply tapered reduction to artifact segments
        for g = 1:length(artifactGroups)
            indices = artifactGroups{g};
            if length(indices) > 10 % Only process significant artifacts
                window = tukeywin(length(indices), 0.3)';
                reduction = 0.3 + 0.7*window; % Reduce to 30-100%
                
                for ch = 1:nChannels
                    eegClean(ch, indices) = eegClean(ch, indices) .* reduction;
                end
            end
        end
    end
    
    % Remove dead channels
    if any(artifacts.deadChannels)
        eegClean = eegClean(~artifacts.deadChannels, :);
    end
end

function eegInterp = interpolateBadChannels(eegData, badChannels)
    % Interpolate bad channels using neighboring channels
    
    [nChannels, nSamples] = size(eegData);
    eegInterp = eegData;
    
    badIdx = find(badChannels);
    goodIdx = find(~badChannels);
    
    if isempty(goodIdx) || isempty(badIdx)
        return;
    end
    
    % Simple nearest neighbor interpolation
    for b = 1:length(badIdx)
        % Find nearest good channel
        [~, nearestIdx] = min(abs(goodIdx - badIdx(b)));
        eegInterp(badIdx(b), :) = eegData(goodIdx(nearestIdx), :);
    end
end