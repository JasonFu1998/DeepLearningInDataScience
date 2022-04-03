rng(400);

book_fname = 'Datasets/tweets2.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);

book_chars = unique(book_data);
K = length(book_chars);

char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
for i = 1: K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end

m = 100;
seq_length = 25; % 10, 40
eta = 0.1;
sig = 0.01;
n_text = 140;
n_epoch = 5;
% n_iter = 20000;
n_iter = floor(n_epoch * length(book_data) / seq_length);


x0 = CharIdx2OneHot(40, K);
h0 = zeros(m, 1);

RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(m, K) * sig;
RNN.W = randn(m, m) * sig;
RNN.V = randn(K, m) * sig;

% gradient check
% X_chars = book_data(1: seq_length);
% Y_chars = book_data(2: seq_length + 1);
% X = Chars2Matrix(X_chars, K, char_to_ind);
% Y = Chars2Matrix(Y_chars, K, char_to_ind);
% ga = ComputeGrad(X, Y, RNN, h0);
% gn = ComputeGradNum(X, Y, RNN, 1e-4);
% W_error = abs(ga.W - gn.W) ./ max(1e-6, abs(ga.W) + abs(gn.W));
% V_error = abs(ga.V - gn.V) ./ max(1e-6, abs(ga.V) + abs(gn.V));
% U_error = abs(ga.U - gn.U) ./ max(1e-6, abs(ga.U) + abs(gn.U));
% b_error = abs(ga.b - gn.b) ./ max(1e-6, abs(ga.b) + abs(gn.b));
% c_error = abs(ga.c - gn.c) ./ max(1e-6, abs(ga.c) + abs(gn.c));

X = Chars2Matrix(book_data, K, char_to_ind);
Y = Chars2Matrix(book_data, K, char_to_ind);
[RNNstar, loss] = Train(RNN, X, Y, h0, seq_length, n_iter, eta, ind_to_char);

text = SynthesizeText(RNNstar, h0, x0, n_text, ind_to_char);
disp(char(text));

function [RNNstar, smooth_loss_list] = Train(RNN, X, Y, h0, seq_length, n_iter, eta, ind_to_char)
    
    e = 1;
    smooth_loss_list = zeros(1, n_iter);
    for f = fieldnames(RNN)'
        m.(f{1}) = zeros(size(RNN.(f{1})));
    end
    
    for i = 1: n_iter
        x = X(:, e: e + seq_length - 1);
        y = Y(:, e + 1: e + seq_length);
        if e == 1
            hprev = h0;
        end
        [grads, hprev, loss] = ComputeGrad(x, y, RNN, hprev);
        
        if i == 1
            smooth_loss = loss;
        end
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
        smooth_loss_list(i) = smooth_loss;
    
        for f = fieldnames(RNN)'
            m.(f{1}) = m.(f{1}) + grads.(f{1}) .^ 2;
            RNN.(f{1}) = RNN.(f{1}) - eta * (grads.(f{1}) ./ (m.(f{1}) + 1e-8) .^ 0.5);
        end
        
        e = e + seq_length;
        if e > size(X, 2) - seq_length - 1
            e = 1;
        end
        
        if i == 1 || mod(i, 10000) == 0
            disp(i);
            disp(smooth_loss);
            text = SynthesizeText(RNN, hprev, x(:, 1), 140, ind_to_char);
            disp(char(text));
        end
    end
    
    RNNstar = RNN;
end

function text = SynthesizeText(RNN, h0, x0, n_text, ind_to_char)
    
    K = size(RNN.c, 1);
    ht = h0;
    xt = x0;
    text = zeros(1, n_text);

    for t = 1: n_text
        a_t = RNN.W * ht + RNN.U * xt + RNN.b;
        h_t = tanh(a_t);
        o_t = RNN.V * h_t + RNN.c;
        p_t = softmax(o_t);
    
        cp = cumsum(p_t);
        a = rand;
        ixs = find(cp - a > 0);
        ii = ixs(1);
        
        ht = h_t;
        xt = CharIdx2OneHot(ii, K);
        text(t) = ind_to_char(ii);
    end
end

function x = CharIdx2OneHot(i, K)

    x = zeros(K, 1);
    x(i, 1) = 1;
end

function M = Chars2Matrix(chars, K, char_to_ind)

    M = zeros(K, length(chars));
    for i = 1: length(chars)
        M(:, i) = CharIdx2OneHot(char_to_ind(chars(i)), K);
    end
end

function [loss, output, P] = ForwardPass(X, Y, RNN, h0)

    loss = 0;
    output = zeros(size(h0, 1), size(X, 2));
    P = zeros(size(X));
    
    ht = h0;
    for t = 1: size(X, 2)
        xt = X(:, t);
        yt = Y(:, t);

        a_t = RNN.W * ht + RNN.U * xt + RNN.b;
        h_t = tanh(a_t);
        o_t = RNN.V * h_t + RNN.c;
        p_t = softmax(o_t);
               
        loss = loss - log(sum(yt .* p_t));
        output(:, t) = h_t;
        P(:, t) = p_t;
     
        ht = h_t;        
    end
end

function [grads, h, loss] = ComputeGrad(X, Y, RNN, h0)

    [loss, H, P] = ForwardPass(X, Y, RNN, h0);
    for f = fieldnames(RNN)'
        grads.(f{1}) = zeros(size(RNN.(f{1})));
    end
    
    for t = size(X, 2): -1: 1
        xt = X(:, t);
        yt = Y(:, t);
        pt = P(:, t);
        ht = H(:, t);

        g_ot = -(yt - pt)';
        grads.c = grads.c + g_ot';
        grads.V = grads.V + g_ot' * ht';
        
        if t == size(X, 2)
            g_ht = g_ot * RNN.V;
        else
            g_ht = g_ot * RNN.V + g_at * RNN.W;
        end
        
        g_at = g_ht * diag(1 - ht .^ 2);
        grads.b = grads.b + g_at';
        
        if t == 1
            h_t = h0;
        else
            h_t = H(:, t - 1);
        end
        
        grads.W = grads.W + g_at' * h_t';
        grads.U = grads.U + g_at' * xt';
    end
    
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    h = H(:, size(X, 2));
end

function num_grads = ComputeGradNum(X, Y, RNN, h)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i = 1: n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ForwardPass(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ForwardPass(X, Y, RNN_try, hprev);
        grad(i) = (l2 - l1) / (2 * h);
    end
end
