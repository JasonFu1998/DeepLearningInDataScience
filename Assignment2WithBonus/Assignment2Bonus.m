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
val.X = trainX(:, 1:5000);
val.Y = trainY(:, 1:5000);
val.y = trainy(1:5000);
train.X = trainX(:, 5001:50000);
train.Y = trainY(:, 5001:50000);
train.y = trainy(5001:50000);

mean_X = mean(train.X, 2);
std_X = std(train.X, 0, 2);
X1 = train.X - repmat(mean_X, [1, size(train.X, 2)]);
train.X = X1 ./ repmat(std_X, [1, size(X1, 2)]);
X2 = val.X - repmat(mean_X, [1, size(val.X, 2)]);
val.X = X2 ./ repmat(std_X, [1, size(X2, 2)]);
X3 = test.X - repmat(mean_X, [1, size(test.X, 2)]);
test.X = X3 ./ repmat(std_X, [1, size(X3, 2)]);

m = 50; % 25 / 50 / 100
d = size(train.X, 1);
K = size(train.Y, 1);
W1 = 1/sqrt(d) * randn(m, d);
W2 = 1/sqrt(m) * randn(K, m);
b1 = zeros(m, 1);
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};

hp.n_batch = 100;
hp.n_epoch = 40;  % 4 epochs equal to 1 cycle
hp.n_s = 2 * floor(size(train.X, 2) / hp.n_batch);
% LR range test
% hp.eta_min = 0;
% hp.eta_max = 0.1;
hp.eta_min = 0.0125;
hp.eta_max = 0.05;
hp.lambda = 0.003143;

% Search(W, b, train, val, test);

[~, ~, model] = MiniBatchGD(train, val, test, W, b, hp);

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
    prob = 0; % probability of each node to be dropped out
    global mask
    mask = (rand(size(h)) >= prob) / (1 - prob);
    h = h .* mask;
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
    s1 = W{1} * X + b{1};
    h = max(0, s1);
    s = W{2} * h + b{2};
    P = softmax(s);
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
    global mask
    G = G .* mask;
    grad_W1 = G * X' / n + 2 * lambda * W{1};
    grad_b1 = G * ones(n, 1) / n;
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};  
end

function acc = ComputeAccuracy(X, y, W, b)
    s1 = W{1} * X + b{1};
    h = max(0, s1);
    s = W{2} * h + b{2};
    P = softmax(s);
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
%     model.train_loss = zeros(1, hp.n_epoch);
%     model.val_loss = zeros(1, hp.n_epoch);
%     model.val_acc = zeros(1, hp.n_epoch);
    model.test_acc = zeros(1, hp.n_epoch);
    
%     model.train_cost = [];
%     model.val_acc = [];
%     model.eta_list = [];    
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
            eta = CyclicLearningRate(iter, hp.eta_min, hp.eta_max, hp.n_s);
            
            W{1} = W{1} - eta * grad_W{1};
            W{2} = W{2} - eta * grad_W{2};
            b{1} = b{1} - eta * grad_b{1};
            b{2} = b{2} - eta * grad_b{2};
            
%             if mod(iter, 10) == 0
%                 model.train_cost(end + 1) = ComputeCost(train.X, train.Y, W, b, hp.lambda);
%                 model.val_acc(end + 1) = ComputeAccuracy(val.X, val.y, W, b);
%                 model.eta_list(end + 1) = eta;
%             end          
        end
        
        model.train_cost(i) = ComputeCost(train.X, train.Y, W, b, hp.lambda);
        model.val_cost(i) = ComputeCost(val.X, val.Y, W, b, hp.lambda); 
%         model.val_acc(i) = ComputeAccuracy(val.X, val.y, W, b);
        model.test_acc(i) = ComputeAccuracy(test.X, test.y, W, b);    
    end
    
    Wstar = W;
    bstar = b;
end

% Adjust to search diffrernt hyper-parameters
function Search(W, b, train, val, test) 
    n = 10;
    % Coarse Search for lambda
    % l_min = -5;
    % l_max = -1;
    
    % Fine Search for lambda
    % l_min = 0.004;
    % l_max = 0.008;
    
    % Search for n_s
    % ns_min = 500;
    % ns_max = 1500;
    
    hp.n_batch = 100;
    % hp.n_epoch = 8;
    % hp.n_s = 2 * floor(size(train.X, 2) / hp.n_batch);
    hp.n_s = 1470;
    hp.eta_min = 1e-5;
    hp.eta_max = 1e-1;
    hp.lambda = 0.004592;
    
    % file = fopen("LambdaCoarseSearch100.txt", "w");
    % file = fopen("LambdaFineSearch100.txt", "w");
    % file = fopen("ns_search.txt", "w");
    file = fopen("cycles_search.txt", "w");
    % fprintf(file, "%d lambda values between l = %d to %d \n\n", n, l_min, l_max);
    % fprintf(file, "%d lambda values between %f and %f \n\n", n, l_min, l_max);
    % fprintf(file, "%d n_s values between %d and %d \n\n", n, ns_min, ns_max);
    fprintf(file, "Number of cycles from 1 to %d \n\n", n);
    
    for i = 1:n
        % l = l_min + (l_max - l_min) * rand(1, 1);
        % Coarse Search for lambda
        % hp.lambda = 10 ^ l;
        % Fine Search for lambda
        % hp.lambda = l;
        
        % Search for the length of cycles
        % hp.n_s = floor(ns_min + (ns_max - ns_min) * rand(1, 1));
        % hp.n_epoch = floor(4 * hp.n_s * hp.n_batch / size(train.X, 2));
        
        % Search for the number of cycles
        hp.n_epoch = i * floor(2 * hp.n_s * hp.n_batch / size(train.X, 2));
        
        [~, ~, model] = MiniBatchGD(train, val, test, W, b, hp);
        vacc = max(model.val_acc);
        tacc = max(model.test_acc);
        
        % fprintf(file, "lambda: %.6f \n", hp.lambda);
        % fprintf(file, "n_s: %d \n", hp.n_s);
        fprintf(file, "%d \n", i);
        fprintf(file, "Validation Accuracy: %.2f \n", vacc * 100);
        fprintf(file, "Test Accuracy: %.2f \n \n", tacc * 100);
    end
    
   fclose(file);
end