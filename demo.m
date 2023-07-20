% clc
% clear all
% close all


%% load data for 2 views
warning off
K = 2; %% the number of view/classes/sources
Xs = cell(1,K);
si = cell(1,K);
tic
%% data preparation
load 2view.mat
Xt1 = NormalizeFea(Xt1);
Xt2 = NormalizeFea(Xt2);
Xs1 = NormalizeFea(Xs1);
Xs2 = NormalizeFea(Xs2);

%% Handling incomplete views
n = min(size(Xs1, 1), size(Xs2, 1));

missing_objs_1 = 1:(n/2);
missing_objs_2 = (n/2):n;

O1 = ones(size(Xs1,1), size(Xs1,2));
O1(missing_objs_1, :) = 0;

O2 = ones(size(Xs2,1), size(Xs2,2));
O2(missing_objs_2, :) = 0;

Xs1 = Xs1 .* O1;
Xs2 = Xs2 .* O2;

Xtt = [Xt1;Xt2]';
Ytt = [Yt1;Yt2];
Xss = [Xs1;Xs2]';
Yss = [Ys1;Ys2];


%% Initialize the data and variable matrices
si{1} = size(Xs1,1);
si{2} = size(Xs2,1);

Xs{1} = Xs1';
Xs{2} = Xs2';
Ys{1} = Ys1;
Ys{2} = Ys2;
options.K = K;



%% dimension for the low-dimensional space
options.ReducedDim = 200;
%% parameter for supervised regularizer
options.lambda3 = 1e1;

%% choose which optimization methods
%% 1 means solution to P without Low-rank constraint
%% 2 menas solution to P with Low-rank constraint
%% 3 means solution to P with Gradient Descent Optimization
options.optP = 3;

Pt = CLRS(Xs,Ys,options);
toc

%% Test Stage
Zs = Pt'*Xss;
Zt = Pt'*Xtt;
Cls = cvKnn(Zt, Zs, Yss, 1);
acc = length(find(Cls==Ytt))/length(Ytt);
fprintf('NN=%0.4f\n',acc);


