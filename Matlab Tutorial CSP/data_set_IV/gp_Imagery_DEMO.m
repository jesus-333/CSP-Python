%% teste gpires
%IMAGINAÇÂO  
% right hand movement vs  feet movement 
% Trials are cut in the interval [0.5–2.5 s] after the onset of the cue.

%% load dos dados e parametros 
% Motor Imagery demo
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

%-----------
%% segmentação em trials
% Trials are cut in the interval [0.5–2.5 s] after the onset of the cue.
% Struct to store the trials in, each class gets an entry
% FORMATO DAS TRIALS
% CANAIS x TimeSamples x TRIALS = 118 x 200 x 140   
% variáveis: estrutura com trials.right e trials.foot
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

%% CÁLCULO DA PSD E PLOT dos DADOS
% cálculo da densidade espectral de potencia
[psd_r, freqs] = calc_psd(trials.right, sample_rate);
[psd_f, freqs] = calc_psd(trials.foot, sample_rate);
trials_PSD = struct('right', psd_r, 'foot', psd_f);

% plot de
%C3: Central, left
%Cz: Central, central
%C4: Central, right
channels_of_interest = [find(strcmp(channel_names, 'C3')), ...
                        find(strcmp(channel_names, 'Cz')), ...
                        find(strcmp(channel_names, 'C4'))];
plot_psds(trials_PSD, freqs, channels_of_interest, ...
         {'left', 'center', 'right'}, 2500, 2);
     
%% FEATURE EXTRACTION: filtering e logPower
% filtragem com butterworth mas com correção de fase usando filtfilt
% frequencias de filtragem 8 a 15 Hz
trials_filt = struct('right', bandpass(trials.right, 8, 15, sample_rate), ...
                     'foot', bandpass(trials.foot, 8, 15, sample_rate));

% plot da densidade espectral de potência dos dados filtrados
[psd_r, freqs] = calc_psd(trials_filt.right, sample_rate);
[psd_f, freqs] = calc_psd(trials_filt.foot, sample_rate);
trials_PSD = struct('right', psd_r, 'foot', psd_f);

plot_psds(trials_PSD, freqs, channels_of_interest, ...
          {'left', 'center', 'right'}, 3000, 2);

%FEATURES
% para cada trial (epoch) é calculada a potência e depois calculado o seu logaritmo
% devolve: channels x trials
trials_logvar = struct('right', logvar(trials_filt.right), ...
                       'foot', logvar(trials_filt.foot));
                   
%plot das médias das variâncias de cada canal   
plot_logvar(trials_logvar); %grafico de barras

%% FILTRO CSP
W = csp(trials_filt.right, trials_filt.foot); %filtro

%filtragem
trials_csp = struct('right', apply_mix(W, trials_filt.right), ...
                    'foot', apply_mix(W, trials_filt.foot));

%potencia das componentes                 
trials_logvar = struct('right', logvar(trials_csp.right), ...
                       'foot', logvar(trials_csp.foot));
                   
%plot das médias das variâncias de cada canal
plot_logvar(trials_logvar); %grafico de barras

[psd_r, freqs] = calc_psd(trials_csp.right, sample_rate);
[psd_f, freqs] = calc_psd(trials_csp.foot, sample_rate);
trials_PSD = struct('right', psd_r, 'foot', psd_f);

plot_psds(trials_PSD, freqs, [1, 59, 118], ...
          {'first component', 'middle component', 'last component'}, 0.75, 2);

%% CLASSIFICAÇÃO
size(trials_logvar.right)  %118 componentes e 140 trials

%plot em duas dimensoes (potencia da primeira componente vs. potencia da última componente)
plot_scatter(trials_logvar.right, trials_logvar.foot)

% separação de trials de TREINO e de TESTE
%Percentage of trials to use for training (50-50 split here)
train_percentage = 0.5;

% Calculate the number of trials for each class the above percentage boils
% down to
ntrain_r = fix(size(trials_filt.right, 3) * train_percentage);
ntrain_f = fix(size(trials_filt.foot, 3) * train_percentage);
ntest_r = size(trials_filt.right, 3) - ntrain_r;
ntest_f = size(trials_filt.foot, 3) - ntrain_f;

% Splitting the frequency filtered signal into a train and test set
train = struct('right', trials_filt.right(:,:,1:ntrain_r), ...
               'foot', trials_filt.foot(:,:,1:ntrain_f));

test = struct('right', trials_filt.right(:,:,ntrain_r+1:end), ...
              'foot', trials_filt.foot(:,:,ntrain_f+1:end));

          
%% TREINO
% Train the CSP on the training set only
W = csp(train.right, train.foot);
% Apply the CSP on both the training and test set
train.right = apply_mix(W, train.right);
train.foot = apply_mix(W, train.foot);
test.right = apply_mix(W, test.right);
test.foot = apply_mix(W, test.foot);
size(train.right)

% Select only the first and last components for classification
comp = [1,118];
train.right = train.right(comp,:,:);
train.foot = train.foot(comp,:,:);
test.right = test.right(comp,:,:);
test.foot = test.foot(comp,:,:);
size(train.right)

% Calculate the log-var feature
train.right = logvar(train.right);
train.foot = logvar(train.foot);
test.right = logvar(test.right);
test.foot = logvar(test.foot);
size(train.right)


%TREINO DO classificador LDA
[W, b] = train_lda(train.right, train.foot);

disp('W:'); disp(W);
disp('b:'); disp(b);

%faz plot da bondary para os DADOS DE TREINO
% Scatterplot like before
plot_scatter(train.right, train.foot);
title('Training data');

% Calculate decision boundary (x,y)
x = (-5:0.1:1);
y = (b - W(1)*x) / W(2);

% Plot the decision boundary
hold on;
plot(x, y, '--k', 'LineWidth', 2);
hold off;
xlim([-5, 1]);
ylim([-2.2, 1]);

% REPETE para os DADOS DE TESTE
plot_scatter(test.right, test.foot);
title('Test data');
hold on;
plot(x, y, '--k', 'LineWidth', 2);
hold off;
xlim([-5, 1]);
ylim([-2.2, 1]);

% aplica LDA e pobtem matriz de confusão
% The number at the diagonal will be trials that were correctly classified,
% any trials incorrectly classified (either a false positive or false
% negative) will be in the corners.
% Print confusion matrix
conf = [sum(apply_lda(test.right, W, b) == 1), sum(apply_lda(test.foot, W, b) == 1); ...
        sum(apply_lda(test.right, W, b) == 2), sum(apply_lda(test.foot, W, b) == 2)];

disp('Confusion matrix:'); disp(conf);
fprintf('\nAccuracy: %.3f\n', sum(diag(conf)) / sum(sum(conf)));