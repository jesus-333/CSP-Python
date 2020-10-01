function X = whitening(sigma)
    % Calculate a whitening matrix for covariance matrix sigma.
    [U, l, ~] = svd(sigma);
    X = U * (l ^ -0.5);
end