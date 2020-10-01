%Motor Imagery demo
m = load('100Hz/data_set_IVa_al.mat');
m2 = load('100Hz/true_labels_al.mat');

sample_rate = m.nfo.fs;
EEG = m.cnt';
nchannels = size(EEG, 1);
nsamples = size(EEG, 2);

channel_names = m.nfo.clab;
event_onsets = m.mrk.pos;
event_codes = m2.true_y;
cl_lab = m.mrk.className;
nclasses = length(cl_lab);
nevents = length(event_onsets);

% Print some information
disp('Shape of EEG:'); disp(size(EEG));
disp('Sample rate:'); disp(sample_rate);
disp('Number of channels:'); disp(nchannels);
disp('Channel names:'); disp(channel_names);
disp('Number of events:'); disp(length(event_onsets));
disp('Event codes:'); disp(unique(event_codes));
disp('Class labels:'); disp(cl_lab);
disp('Number of classes:'); disp(nclasses);

%%

% Struct to store the trials in, each class gets an entry
trials = struct();

% The time window (in samples) to extract for each trial, here 0.5 -- 2.5
% seconds
win = fix(0.5*sample_rate):fix(2.5*sample_rate)-1;

% Length of the time window
nsamples = length(win);

% Loop over the classes (right, foot)
codes = unique(event_codes);
for i = 1:length(cl_lab)
    cl = cl_lab{i};
    code = codes(i);

    % Extract the onsets for the class
    cl_onsets = event_onsets(event_codes == code);

    % Allocate memory for the trials
    trials.(cl) = zeros(nchannels, nsamples, length(cl_onsets));

    % Extract each trial
    for j = 1:length(cl_onsets)
        onset = cl_onsets(j);
        trials.(cl)(:,:,j) = EEG(:, win+onset);
    end
end

% Some information about the dimensionality of the data (channels x time x trials)
disp('Shape of trials.right:'); disp(size(trials.right));
disp('Shape of trials.foot:'); disp(size(trials.foot));
