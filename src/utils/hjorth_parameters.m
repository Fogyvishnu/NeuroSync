function [activity, mobility, complexity] = hjorth_parameters(signal)
    % HJORTH_PARAMETERS - Calculate Hjorth parameters
    %
    % Inputs:
    %   signal - Time series data
    %
    % Outputs:
    %   activity - Variance of the signal
    %   mobility - sqrt(var(dx/dt) / var(x))
    %   complexity - mobility(dx/dt) / mobility(x)
    
    % Activity: variance of the signal
    activity = var(signal);
    
    % First derivative
    diff1 = diff(signal);
    
    % Mobility: sqrt(var(dx/dt) / var(x))
    if activity > 0
        mobility = sqrt(var(diff1) / activity);
    else
        mobility = 0;
    end
    
    % Second derivative
    diff2 = diff(diff1);
    
    % Complexity: mobility(dx/dt) / mobility(x)
    if mobility > 0 && var(diff1) > 0
        complexity = sqrt(var(diff2) / var(diff1)) / mobility;
    else
        complexity = 0;
    end
end