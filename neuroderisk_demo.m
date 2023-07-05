% clc
% clear all
% close all


%% load data for 2 views
warning off
K = 5; %% the number of view/classes/sources
Xs = cell(1,K);
si = cell(1,K);

%% data preparation
load neuroderisk.mat
Xs{1} = uca;
Xs{2} = novartis;
Xs{3} = sard;
Xs{4} = unifi;
Xs{5} = msd;

G = cell(1,K);
G{1} = G_uca;
G{2} = G_novartis;
G{3} = G_sard;
G{4} = G_unifi;
G{5} = G_msd;

for i=1:K
    Xs{i} = NormalizeFea(Xs{i});
    Xs{i} = Xs{i} * G{i}';
end


%% Initialize the data and variable matrices

Xss = [];
Yss = Compound';

%Ys{1} = Y_uca;
%Ys{2} = Y_novartis;
%Ys{3} = Y_sard;
%Ys{4} = Y_unifi;
%Ys{5} = Y_msd;

for i=1:K
    Xss = [Xss; Xs{i}];
    %Yss = [Yss; Ys{i}];
end
Xss = Xss';
Xtt = Xss;
Ytt = Yss;


for i=1:K
    Xs{i} = Xs{i}';
end

options.K = K;

%% dimension for the low-dimensional space
options.ReducedDim = 150;
%% parameter for supervised regularizer
options.lambda3 = 1e1;

%% choose which optimization methods
%% 1 means solution to P without Low-rank constraint
%% 2 means solution to P with Low-rank constraint
%% 3 means solution to P with Gradient Descent Optimization
options.optP = 2;

Pt = Copy_of_CLRS(Xs,Ys,options);

%% Test Stage
Zs = Pt'*Xss;
Zt = Pt'*Xtt;
Cls = cvKnn(Zt, Zs, Yss, 1);
acc = length(find(Cls==Ytt))/length(Ytt);
fprintf('NN=%0.4f\n',acc);


