rng(400);

[trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
[trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
[trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
[trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
[trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
[test.X, test.Y, test.y] = LoadBatch('test_batch.mat');
trainX = [trainX1, trainX2, trainX3, trainX4, trainX5];
trainY = [trainY1, trainY2, trainY3, trainY4, trainY5];
trainy = [trainy1, trainy2, trainy3, trainy4, trainy5];
val.X = trainX(:, 1:1000);
val.Y = trainY(:, 1:1000);
val.y = trainy(1:1000);
train.X = trainX(:, 1001:50000);
train.Y = trainY(:, 1001:50000);
train.y = trainy(1001:50000);

% [train.X, train.Y, train.y] = LoadBatch('data_batch_3.mat');
% [val.X, val.Y, val.y] = LoadBatch('data_batch_4.mat');
% [test.X, test.Y, test.y] = LoadBatch('test_batch.mat');

% train.X = train.X(:, 1:100);
% train.Y = train.Y(:, 1:100);
% train.y = train.y(1:100);

mean_X = mean(train.X, 2);
std_X = std(train.X, 0, 2);
X1 = train.X - repmat(mean_X, [1, size(train.X, 2)]);
train.X = X1 ./ repmat(std_X, [1, size(X1, 2)]);
X2 = val.X - repmat(mean_X, [1, size(val.X, 2)]);
val.X = X2 ./ repmat(std_X, [1, size(X2, 2)]);
X3 = test.X - repmat(mean_X, [1, size(test.X, 2)]);
test.X = X3 ./ repmat(std_X, [1, size(X3, 2)]);

m = 50;
d = size(train.X, 1);
K = size(train.Y, 1);
W1 = 1/sqrt(d) * randn(m, d);
W2 = 1/sqrt(m) * randn(K, m);
b1 = zeros(m, 1);
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};

hp.n_batch = 100;
hp.n_epoch = 24;
hp.n_s = 1470;
hp.eta_min = 1e-5;
hp.eta_max = 1e-1;
hp.lambda = 0.003143;

% Gradient Check
% X_check = train.X(1:20, 1:2);
% Y_check = train.Y(:, 1:2);
% W_check = {W{1}(:, 1:20), W{2}};
% [P, H] = EvaluateClassifier(X_check, W_check, b);
% [ga_W, ga_b] = ComputeGradients(X_check, Y_check, P, H, W_check, hp.lambda);
% [gn_b, gn_W] = ComputeGradsNumSlow(X_check, Y_check, W_check, b, hp.lambda, 1e-5);
% W1_relative = abs(ga_W{1} - gn_W{1}) ./ max(1e-6, abs(ga_W{1}) + abs(gn_W{1}));
% W2_relative = abs(ga_W{2} - gn_W{2}) ./ max(1e-6, abs(ga_W{2}) + abs(gn_W{2}));
% b1_relative = abs(ga_b{1} - gn_b{1}) ./ max(1e-6, abs(ga_b{1}) + abs(gn_b{1}));
% b2_relative = abs(ga_b{2} - gn_b{2}) ./ max(1e-6, abs(ga_b{2}) + abs(gn_b{2}));

[Wstar, bstar, model] = MiniBatchGD(train, val, test, W, b, hp);

% Search(W, b, train, val, test);

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

function [P, h] = EvaluateClassifier(X, W, b)
    s1 = W{1} * X + b{1};
    h = max(0, s1);
    s = W{2} * h + b{2};
    P = softmax(s);
end

function eta = CyclicLearningRate(i, eta_min, eta_max, n_s)
    t = mod(i, 2 * n_s) / n_s;
    if t < 1
       eta = eta_min + t * (eta_max - eta_min); 
    else
       eta = eta_max - (t - 1) * (eta_max - eta_min);
    end
end

function [cost, loss] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    loss = - sum(log(sum(Y .* P))) / size(X, 2);
    cost = loss + lambda * sumsqr(W{1}) + lambda * sumsqr(W{2});
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, H, W, lambda)
    n = size(X, 2);
    G = - (Y - P);
    grad_W2 = G * H' / n + 2 * lambda * W{2};
    grad_b2 = G * ones(n, 1) / n;
    G = W{2}' * G;
    H(H < 0) = 0;
    H(H > 0) = 1;
    G = G .* H;
    grad_W1 = G * X' / n + 2 * lambda * W{1};
    grad_b1 = G * ones(n, 1) / n;
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};  
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

function [Wstar, bstar, model] = MiniBatchGD(train, val, test, W, b, hp)

    model.train_cost = zeros(1, hp.n_epoch);
    model.val_cost = zeros(1, hp.n_epoch);
    model.train_loss = zeros(1, hp.n_epoch);
    model.val_loss = zeros(1, hp.n_epoch);
    model.test_acc = zeros(1, hp.n_epoch);
    % model.val_acc = zeros(1, hp.n_epoch);
    iter = 0;
    
    for i = 1:hp.n_epoch
        idx = randperm(size(train.X, 2));
        train.X = train.X(:, idx);
        train.Y = train.Y(:, idx);
        train.y = train.y(idx);
        for j = 1:(size(train.X, 2) / hp.n_batch)
            iter = iter + 1;
            j_start = (j - 1) * hp.n_batch + 1;
            j_end = j * hp.n_batch;
            inds = j_start:j_end;
            Xbatch = train.X(:, inds);
            Ybatch = train.Y(:, inds);
            
            [P, H] = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, H, W, hp.lambda);          
            % eta = 0.001;
            eta = CyclicLearningRate(iter, hp.eta_min, hp.eta_max, hp.n_s);
            W{1} = W{1} - eta * grad_W{1};
            W{2} = W{2} - eta * grad_W{2};
            b{1} = b{1} - eta * grad_b{1};
            b{2} = b{2} - eta * grad_b{2};
            
        end
        [model.train_cost(i), model.train_loss(i)] = ComputeCost(train.X, train.Y, W, b, hp.lambda);
        [model.val_cost(i), model.val_loss(i)] = ComputeCost(val.X, val.Y, W, b, hp.lambda); 
        % model.val_acc(i) = ComputeAccuracy(val.X, val.y, W, b);
        model.test_acc(i) = ComputeAccuracy(test.X, test.y, W, b);
    end
    Wstar = W;
    bstar = b;
end

function Search(W, b, train, val, test)
    n = 10;
    l_min = 0.001;
    l_max = 0.005;
    
    hp.n_batch = 100;
    hp.n_epoch = 8;
    hp.n_s = 2 * floor(size(train.X, 2) / hp.n_batch);
    hp.eta_min = 1e-5;
    hp.eta_max = 1e-1;
    file = fopen("LambdaFineSearch.txt", "w");
    fprintf(file, "%d lambda values between %d and %d \n\n", n, l_min, l_max);
    for i = 1:n
        l = l_min + (l_max - l_min) * rand(1, 1);
        hp.lambda = l;  
        [~, ~, model] = MiniBatchGD(train, val, test, W, b, hp);
        acc = max(model.val_acc);
        fprintf(file, "lambda: %.6f \n", hp.lambda);
        fprintf(file, "Validation Accuracy: %.2f \n \n", acc * 100);
    end
   fclose(file);
end


function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
    
        for i=1:length(b{j})
        
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
        
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
        
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
    
        for i=1:numel(W{j})
        
            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
    
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);
    
            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end
