rng(400);

[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);
X1 = trainX - repmat(mean_X, [1, size(trainX, 2)]);
trainX = X1 ./ repmat(std_X, [1, size(X1, 2)]);
X2 = valX - repmat(mean_X, [1, size(valX, 2)]);
valX = X2 ./ repmat(std_X, [1, size(X2, 2)]);
X3 = testX - repmat(mean_X, [1, size(testX, 2)]);
testX = X3 ./ repmat(std_X, [1, size(X3, 2)]);

W = 0.01 * randn(10, 3072);
b = 0.01 * randn(10, 1);

n_batch = 100;
eta = 0.001;
n_epoch = 40;
lambda = 1;

% Gradient Check
% X_check = trainX(1:20, 1);
% Y_check = trainY(:, 1);
% W_check = W(:, 1:20);
% P = EvaluateClassifier(X_check, W_check, b);
% [ga_W, ga_b] = ComputeGradients(X_check, Y_check, P, W_check, lambda);
% [gn_b, gn_W] = ComputeGradsNum(X_check, Y_check, W_check, b, lambda, 1e-6);
% [gn_b, gn_W] = ComputeGradsNumSlow(X_check, Y_check, W_check, b, lambda, 1e-6);
% W_relative = abs(ga_W - gn_W) ./ max(1e-6, abs(ga_W) + abs(gn_W));
% b_relative = abs(ga_b - gn_b) ./ max(1e-6, abs(ga_b) + abs(gn_b));

[Wstar, bstar, train_loss, val_loss, test_acc] = ...
    MiniBatchGD(trainX, trainY, valX, valY, testX, testy, ...
    n_batch, eta, n_epoch, W, b, lambda);

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [1,10]);

function [X, Y, y] = LoadBatch(file)
    A = load(file);
    X = im2double(A.data');
    y = A.labels;
    Y = zeros(size(y, 1), 10);
    for i = 1:size(y, 1)
        for j = 1:10
            if j == y(i) + 1
                Y(i, j) = 1;
            end
        end
    end
    Y = Y';
end

function P = EvaluateClassifier(X, W, b)
    s = W * X + b;
    P = softmax(s);
end

function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    J = - sum(log(sum(Y .* P))) / size(X, 2) + lambda * sumsqr(W);
end

function l = SVMLoss(X, Y, W, b, lambda)
    l = 0;
    for i = 1:size(X, 2)
        [~, y] = max(Y(:, i));
        for j = 1:size(W, 1)            
            l = l + max(0, W(j, :) * X(:, i) + b(j) - ...
                W(y, :) * X(:, i) - b(y) + 1);               
        end
        l = l - 1;
    end
    l = l / size(X, 2) + lambda * sumsqr(W);
end

function [grad_W, grad_b] = SVMgradients(X, Y, W, b, lambda)       
    grad_W = zeros(size(W));
    grad_b = zeros(size(W, 1), 1);
    for i = 1:size(X, 2)
        gW = zeros(size(W));
        gb = zeros(size(W, 1), 1);
        [~, y] = max(Y(:, i));
        for j = 1:size(W, 1)
            if j ~= y 
                if W(j, :) * X(:, i) + b(j) - W(y, :) * X(:, i) - b(y) + 1 > 0
                    gW(j, :) = X(:, i)';
                    gb(j, 1) = 1;
                end
            end   
        end
        gW(y, :) = -sum(gW);
        gb(y, 1) = -sum(gb);
        grad_W = grad_W + gW;
        grad_b = grad_b + gb;
    end
    grad_W = grad_W / size(X, 2) + 2 * lambda * W;
    grad_b = grad_b / size(X, 2);
end

function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    num = 0;
    for i = 1:size(X, 2)
        [~, I] = max(P(:, i));
        if I == y(i) + 1
            num = num + 1;
        end
    end
    acc = num / size(X, 2);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    n = size(X, 2);
    G = - (Y - P);
    grad_W = G * X' / n + 2 * lambda * W;
    grad_b = G * ones(n, 1) / n;
end

function [Wstar, bstar, train_loss, val_loss, test_acc] = ...
    MiniBatchGD(trainX, trainY, valX, valY, testX, testy, ...
    n_batch, eta, n_epoch, W, b, lambda)

    train_loss = zeros(1, n_epoch);
    val_loss = zeros(1, n_epoch);
    test_acc = zeros(1, n_epoch);
    
    for i = 1:n_epoch
        idx = randperm(size(trainX, 2));
        trainX = trainX(:, idx);
        trainY = trainY(:, idx);
        for j = 1:(size(trainX, 2) / n_batch)
            j_start = (j - 1) * n_batch + 1;
            j_end = j * n_batch;
            inds = j_start:j_end;
            Xbatch = trainX(:, inds);
            Ybatch = trainY(:, inds);
            
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            % [grad_W, grad_b] = SVMgradients(Xbatch, Ybatch, W, b, lambda);
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        train_loss(i) = ComputeCost(trainX, trainY, W, b, lambda);
        val_loss(i) = ComputeCost(valX, valY, W, b, lambda);
        % train_loss(i) = SVMLoss(trainX, trainY, W, b, lambda);
        % val_loss(i) = SVMLoss(valX, valY, W, b, lambda);        
        test_acc(i) = ComputeAccuracy(testX, testy, W, b);
    end
    Wstar = W;
    bstar = b;
end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    c = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)   
    
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W(i) = (c2-c) / h;
    end

end


function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)
    
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W(i) = (c2-c1) / (2*h);
    end
    
end