rng(4);

[trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
[trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
[trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
[trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
[trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
trainX = [trainX1, trainX2, trainX3, trainX4, trainX5];
trainY = [trainY1, trainY2, trainY3, trainY4, trainY5];
trainy = [trainy1, trainy2, trainy3, trainy4, trainy5];
val.X = trainX(:, 1:5000);
val.Y = trainY(:, 1:5000);
val.y = trainy(1:5000);
train.X = trainX(:, 5001:50000);
train.Y = trainY(:, 5001:50000);
train.y = trainy(5001:50000);
[test.X, test.Y, test.y] = LoadBatch('test_batch.mat');

mean_X = mean(train.X, 2);
std_X = std(train.X, 0, 2);
X1 = train.X - repmat(mean_X, [1, size(train.X, 2)]);
train.X = X1 ./ repmat(std_X, [1, size(X1, 2)]);
X2 = val.X - repmat(mean_X, [1, size(val.X, 2)]);
val.X = X2 ./ repmat(std_X, [1, size(X2, 2)]);
X3 = test.X - repmat(mean_X, [1, size(test.X, 2)]);
test.X = X3 ./ repmat(std_X, [1, size(X3, 2)]);

d = size(train.X, 1);
K = size(train.Y, 1);
% m = [d, 50, 50, K];
% m = [d, 500, 100, K];
% m = [d, 50, 40, 30, 20, 10, K];
% m = [d, 500, 250, 100, 50, 20, K];
m = [d, 1000, 500, 250, 100, 30, K];
% m = [d, 50, 30, 20, 20, 10, 10, 10, 10, K];
% m = [d, 500, 250, 100, 50, 40, 30, 20, 10, K];
[NP.W, NP.b, NP.gamma, NP.beta] = ParameterInitialize(m);

hp.n_batch = 100;
hp.n_epoch = 20;
hp.n_s = 2250;
hp.eta_min = 1e-5;
hp.eta_max = 1e-1;
hp.lambda = 0.003819;
hp.alpha = 0.7;

% Search(NP, train, val, test);

model = MiniBatchGD(train, val, test, NP, hp);

function [W, b, gamma, beta] = ParameterInitialize(m)
    k = size(m, 2) - 1;
    W = cell(1, k);
    b = cell(1, k);
    gamma = cell(1, k - 1);
    beta = cell(1, k - 1);
    for i = 1: k - 1
        W{i} = 1 / sqrt(m(i)) * randn(m(i + 1), m(i));
        b{i} = zeros(m(i + 1), 1);
        gamma{i} = ones(m(i + 1), 1);
        beta{i} = zeros(m(i + 1), 1);
    end
    W{k} = 1 / sqrt(m(k)) * randn(m(k + 1), m(k));
    b{k} = zeros(m(k + 1), 1); 
end

function [X, Y, y] = LoadBatch(file)
    A = load(file);
    X = im2double(A.data');
    y = A.labels;
    Y = zeros(size(y, 1), 10);
    for i = 1: size(y, 1)
        for j = 1: 10
            if j == y(i) + 1
                Y(i, j) = 1;
            end
        end
    end
    Y = Y';
end

function [P, H] = EvaluateClassifier(X, NP, mu, v)
    k = size(NP.W, 2);
    H.mu = cell(1, k - 1);
    H.v = cell(1, k - 1);
    H.a = cell(1, k - 1);
    H.b = cell(1, k - 1);
    H.c = cell(1, k);
    H.c{1} = X;
    
    for i = 1: k - 1
        % H.a{i} = NP.W{i} * H.c{i} + NP.b{i};
        H.a{i} = max(0, NP.W{i} * H.c{i} + NP.b{i});
        if nargin == 4
            H.b{i} = diag((v{i} + eps) .^ -0.5) * (H.a{i} - mu{i});
        else
            H.mu{i} = mean(H.a{i}, 2);
            H.v{i} = var(H.a{i}, 1, 2);
            H.b{i} = diag((H.v{i} + eps) .^ -0.5) * (H.a{i} - H.mu{i});
        end 
        % H.c{i + 1} = max(0, NP.gamma{i} .* H.b{i} + NP.beta{i});
        H.c{i + 1} = NP.gamma{i} .* H.b{i} + NP.beta{i};
    end
    P = softmax(NP.W{k} * H.c{k} + NP.b{k});
end

function eta = CyclicLearningRate(i, eta_min, eta_max, n_s)
    t = mod(i, 2 * n_s) / n_s;
    if t < 1
       eta = eta_min + t * (eta_max - eta_min); 
    else
       eta = eta_max - (t - 1) * (eta_max - eta_min);
    end
end

function [cost, loss] = ComputeCost(X, Y, NP, lambda, mu, v)
    if nargin == 6
        P = EvaluateClassifier(X, NP, mu, v);
    else
        P = EvaluateClassifier(X, NP);
    end
    loss = - sum(log(sum(Y .* P))) / size(X, 2);
    cost = 0;
    for i = 1: size(NP.W, 2)
        cost = cost + lambda * sumsqr(NP.W{i});
    end
    cost = loss + cost;
end

function grad = ComputeGradients(X, Y, P, H, NP, lambda)
    k = size(NP.W, 2);
    grad.W = cell(1, k);
    grad.b = cell(1, k);
    grad.gamma = cell(1, k - 1);
    grad.beta = cell(1, k - 1);
    n = size(X, 2);
    G = - (Y - P);
    grad.W{k} = G * H.c{k}' / n + 2 * lambda * NP.W{k};
    grad.b{k} = G * ones(n, 1) / n;
    G = NP.W{k}' * G;
    % G = G .* (H.c{k} > 0);
    for i = k - 1: -1: 1
        grad.gamma{i} = (G .* H.b{i}) * ones(n, 1) / n;
        grad.beta{i} = G * ones(n, 1) / n;
        G = G .* (NP.gamma{i} * ones(1, n));
        G = BatchNormBackPass(G, H.a{i}, H.mu{i}, H.v{i});
        G = G .* (H.a{i} > 0);
        grad.W{i} = G * H.c{i}' / n + 2 * lambda * NP.W{i};
        grad.b{i} = G * ones(n, 1) / n;
        G = NP.W{i}' * G;
        % G = G .* (H.c{i} > 0);
    end
end

function Gb = BatchNormBackPass(G, S, mu, v)
    s1 = (v + eps) .^ -0.5;
    s2 = (v + eps) .^ -1.5;
    n = size(G, 2);
    G1 = G .* (s1 * ones(1, n));
    G2 = G .* (s2 * ones(1, n));
    D = S - mu * ones(1, n);
    c = (G2 .* D) * ones(n, 1);
    Gb = G1 - G1 * ones(n, 1) * ones(1, n) / n - D .* (c * ones(1, n)) / n;
end

function acc = ComputeAccuracy(X, y, NP, mu, v)
    P = EvaluateClassifier(X, NP, mu, v);
    num = 0;
    for i = 1: size(X, 2)
        [~, I] = max(P(:, i));
        if I == y(i) + 1
            num = num + 1;
        end
    end
    acc = num / size(X, 2);
end

function model = MiniBatchGD(train, val, test, NP, hp)

    % model.train_cost = zeros(1, hp.n_epoch);
    % model.val_cost = zeros(1, hp.n_epoch);
    model.val_acc = zeros(1, hp.n_epoch);
    model.test_acc = zeros(1, hp.n_epoch);
    
    iter = 0;
    k = size(NP.W, 2);
    for i = 1: hp.n_epoch
        idx = randperm(size(train.X, 2));
        train.X = train.X(:, idx);
        train.Y = train.Y(:, idx);
        train.y = train.y(idx);
        
        for j = 1: (size(train.X, 2) / hp.n_batch)
            iter = iter + 1;
            j_start = (j - 1) * hp.n_batch + 1;
            j_end = j * hp.n_batch;
            inds = j_start: j_end;
            Xbatch = train.X(:, inds);
            Ybatch = train.Y(:, inds);
            
            [P, H] = EvaluateClassifier(Xbatch, NP);
            grad = ComputeGradients(Xbatch, Ybatch, P, H, NP, hp.lambda);
            eta = CyclicLearningRate(iter, hp.eta_min, hp.eta_max, hp.n_s);
            for n = 1: k - 1
                NP.W{n} = NP.W{n} - eta * grad.W{n};
                NP.b{n} = NP.b{n} - eta * grad.b{n};
                NP.gamma{n} = NP.gamma{n} - eta * grad.gamma{n};
                NP.beta{n} = NP.beta{n} - eta * grad.beta{n};     
            end
            NP.W{k} = NP.W{k} - eta * grad.W{k};
            NP.b{k} = NP.b{k} - eta * grad.b{k};
            
            if j == 1
                mu_av = H.mu;
                v_av = H.v;
            else
                for c = 1: k - 1 
                    mu_av{c} = hp.alpha * mu_av{c} + (1 - hp.alpha) * H.mu{c};
                    v_av{c} = hp.alpha * v_av{c} + (1 - hp.alpha) * H.v{c};
                end
            end
        end
        % model.train_cost(i) = ComputeCost(train.X, train.Y, NP, hp.lambda, mu_av, v_av);
        % model.val_cost(i) = ComputeCost(val.X, val.Y, NP, hp.lambda, mu_av, v_av);
        model.val_acc(i) = ComputeAccuracy(val.X, val.y, NP, mu_av, v_av);
        model.test_acc(i) = ComputeAccuracy(test.X, test.y, NP, mu_av, v_av);
        
    end

end

function Search(NP, train, val, test)
    n = 10;
    l_min = 0.003;
    l_max = 0.007;
    hp.alpha = 0.7;
    hp.n_batch = 100;
    hp.n_epoch = 8;
    hp.n_s = 900;
    hp.eta_min = 1e-5;
    hp.eta_max = 1e-1;
    file = fopen("LambdaSearchBonus.txt", "w");
    fprintf(file, "%d lambda values between %.6f and %.6f \n \n", n, l_min, l_max);
    for i = 1: n
        l = l_min + (l_max - l_min) * rand(1, 1);
        hp.lambda = l;
        model = MiniBatchGD(train, val, test, NP, hp);
        acc = max(model.val_acc);
        fprintf(file, "lambda: %.6f \n", hp.lambda);
        fprintf(file, "Validation Accuracy: %.2f \n \n", acc * 100);
    end
   fclose(file);
end
