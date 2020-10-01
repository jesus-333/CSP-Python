function W = csp(trials_r, trials_f)
%     Calculate the CSP transformation matrix W.
%     arguments:
%         trials_r - Array (channels x samples x trials) containing right hand movement trials
%         trials_f - Array (channels x samples x trials) containing foot movement trials
%     returns:
%         Mixing matrix W
    cov_r = trial_cov(trials_r);
    cov_f = trial_cov(trials_f);
    P = whitening(cov_r + cov_f);
    [B,~,~] = svd(P' * cov_f * P);
    W = P * B;
end