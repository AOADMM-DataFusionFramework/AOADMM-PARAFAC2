%% script_test_AOADMM PARAFAC2 (Frobenius loss only) with missing data, using EM approach
close all
clear all
%% add AO-ADMM solver functions to path
addpath(genpath('.\functions_PARAFAC2'))
addpath(genpath('.\data_for_test_scripts'))
%% add other apckages to your path!
addpath(genpath('...\tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
addpath(genpath('...\proximal_operators\code\matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
%% plot configurations
set(0,'DefaultAxesFontSize',22)
set(0,'DefaultLineLineWidth',4)
set(0,'DefaultTextFontSize',22)
set(0,'DefaultLineMarkerSize',2) %Marker size
%% create PARAFAC2 data (here Shift PARAFAC)
sz_A = 20; %I
sz_C = 20; %K
sz_B = 30*ones(1,sz_C); %J_k
R = 3; %number of components

noise_level = 0.2;

K = sz_C;

A = randn(sz_A,R);
%A(A<0) = 0;
C = rand(sz_C,R)+0.1;
B{1} = randn(sz_B(1),R);
%B{1}(B{1}<0) = 0;
for r=1:R %normalize columnwise
    A(:,r) = A(:,r)/norm(A(:,r),2);
    C(:,r) = C(:,r)/norm(C(:,r),2);
    B{1}(:,r) = B{1}(:,r)/norm(B{1}(:,r),2);
end
%SHIFT PARAFAC
for k=2:K
    B{k} = circshift(B{1},k-1);
end
%construct tensor slices
for k=1:K
    X{k} = A*diag(C(k,:))*B{k}';
end
% add noise 
for k=1:K
    normXk = norm(X{k},'fro');
    Nk = randn(size(X{k}));
    normNk = norm(Nk,'fro');
    X{k} = X{k} + noise_level*normXk/normNk*Nk;
end
%% create Z.object 

for k=1:K
    Z.object{k} = X{k};
end
%% specify missing entries (logical matrices, where 0 indicates a missing entry)
load ('missing_mask.mat')
for k=1:K
    Z.miss{k} = missing_mask{k};
end
%% impute missing entries
for k=1:K
    Z.original_object{k} = Z.object{k};
    for j=1:size(Z.object{k},2)
        Z.object{k}(~Z.miss{k}(:,j),j) = 0;%mean(Z.object{k}(Z.miss{k}(:,j),j)); % imput with column-mean
    end
end
%% Parameters for AOADMM
constrained_modes = [0 0 1]; % for modes [A,B,C]: 1 if the mode is constrained in some way, 0 otherwise!
constraints = cell(length(constrained_modes),1); % cell array of length number of modes containing the type of constraint or regularization for each mode, empty if no constraint
%specify constraints-regularizations for each mode, find the options in the file "List of constraints and regularizations.txt"

constraints{3} = {'non-negativity'};

%% add optional ridge regularization performed via primal variable updates, not proximal operators (for no ridge leave field empty), will automatically be added to function value computation
%Z.ridge = [1e-3,1e-3,1e-3]; % penalties for modes [A,Bk,C]
%%

Z.size  = {sz_A,sz_B,sz_C};
Z.constrained_modes = constrained_modes;
Z.constraints = constraints;
Z.R = R;

%% add optional ridge regularization performed via primal variable updates, not proximal operators (for no ridge leave field empty)
%Z.optional_ridge_penalties = [1e-3,1e-3,1e-3]; % penalties for modes [A,Bk,C]
%% Create random initialization
[prox_operators,reg_func] = constraints_to_prox(Z.constrained_modes,Z.constraints,Z.size); 

init.A = randn(sz_A,R);
init.C = rand(sz_C,R);

init.DeltaB = randn(R,R);
for k=1:K
    init.mu_DeltaB{k} = randn(sz_B(k),R);
end

for k=1:K
    init.B{k} = randn(sz_B(k),R);
    init.P{k} = eye(sz_B(k),R);
end
if constrained_modes(1)
    init.ZA = rand(sz_A,R);
    init.ZA = feval(prox_operators{1},init.ZA,1);
    init.mu_A = rand(sz_A,R);
end
if constrained_modes(3)
    init.ZC = rand(sz_C,R);
    init.ZC = feval(prox_operators{3},init.ZC',1); %row-wise!
    init.ZC = init.ZC';
    init.mu_C = rand(sz_C,R);
end
if constrained_modes(2)
    for k=1:K
        init.ZB{k} = rand(sz_B(k),R);
        init.ZB{k} = feval(prox_operators{2},init.ZB{k},1);
        init.mu_B_Z{k} = rand(sz_B(k),R);
    end
end
%% Fit using AOADMM  

options.Display ='iter'; %  set to 'iter' or 'final' or 'no'
options.DisplayIters = 100;
options.MaxOuterIters = 2000;
options.MaxInnerIters = 5;
options.AbsFuncTol   = 1e-7;
options.OuterRelTol = 1e-8;
options.innerRelPrTol_coupl = 1e-5;
options.innerRelPrTol_constr = 1e-5;
options.innerRelDualTol_coupl = 1e-5;
options.innerRelDualTol_constr = 1e-5;

%% set additional (optional) options for diificult constraints/regularizations on mode Bk of PARAFAC2
%options.iter_start_Bkconstraint = 100; % set the number of iterations after which the constraint on Bk will be active 
%options.increase_factor_rhoBk = 10; % set the factor by which the automatically selected value of rho_Bk is increased
%% fit the model

fprintf('AOADMM PARAFAC2 \n')
tic
[FacSol,FacInit,out] = PAR2_AOADMM_EM(Z,options,init); 
toc

%% FMS, normalization and permutation
% in normSol, factor matrices A and Bk are normalized columnwise, the
% scaling is moved to factor matrix C, all factors are permute to best
% match the original factors
[FMS_A,normSol.A,FLAG_A,PERM_A] = score(ktensor(ones(R,1),FacSol.A),ktensor(ones(R,1),A),'lambda_penalty',false);
normSol.A = normSol.A.U{1};

largeB = [];
SollargeB = [];
for k=1:K
    [FMS_Bk{k},normSol.B{k}] = score(ktensor(ones(R,1),FacSol.B{k}),ktensor(ones(R,1),B{k}),'lambda_penalty',false);
    normSol.B{k} = normSol.B{k}.U{1};
    SollargeB = [SollargeB;FacSol.B{k}];
    largeB = [largeB;B{k}];
end
[FMS_B,~] = score(ktensor(ones(R,1),SollargeB),ktensor(ones(R,1),largeB),'lambda_penalty',false);

normSol.C = FacSol.C;
for r=1:R
    normSol.C(:,r) = normSol.C(:,r).*norm(FacSol.A(:,r),2); %move the norm scaling to C
    for k=1:K
        normSol.C(k,r) = normSol.C(k,r)*norm(FacSol.B{k}(:,r),2);
    end
end
normSol.C = normSol.C(:,PERM_A); %permute columns
[FMS_C,~] = score(ktensor(ones(R,1),FacSol.C),ktensor(ones(R,1),C),'lambda_penalty',false); %FMS
%% plot factors
figure()
for r=1:R
    subplot(1,R,r)
    plot(A(:,r))
    hold on
    plot(normSol.A(:,r),'--')
end
legend('true','estimated')
sgtitle('factor matrix A','FontSize',30)

figure()
for r=1:R
    subplot(1,R,r)
    plot(C(:,r))
    hold on
    plot(normSol.C(:,r),'--')
end
legend('true','estimated')
sgtitle('factor matrix C','FontSize',30)

kk=1;
figure()
for r=1:R
    subplot(1,R,r)
    plot(B{kk}(:,r))
    hold on
    plot(normSol.B{kk}(:,r),'--')
end
legend('true','estimated')
sgtitle(['factor matrix B',num2str(kk)],'FontSize',30)

%% convergence
figure()
subplot(1,3,1)
semilogy([0:out.OuterIterations],out.func_val_conv)
hold on
semilogy([0:out.OuterIterations],out.func_coupl_conv,'--')
hold on
semilogy([0:out.OuterIterations],out.func_constr_conv,':')
xlabel('iterations')
ylabel('function value')
legend('function value','difference coupling','difference constraints')


subplot(1,3,2)
semilogy(out.time_at_it,out.func_val_conv)
hold on
semilogy(out.time_at_it,out.func_coupl_conv,'--')
hold on
semilogy(out.time_at_it,out.func_constr_conv,':')
xlabel('time in seconds')
ylabel('function value')
legend('function value','difference coupling','difference constraints')

markers = {'+','o','*','x','^','v','s','d','>','<','p','h'};
subplot(1,3,3)
for i=1:3
    plot(out.innerIters(i,:),markers{i})
    hold on
end
xlabel('outer iteration')
ylabel('inner iterations')
legend('mode 1', 'mode 2','mode 3')
sgtitle('convergence','FontSize',30)

%% missing data estimation performance
TCSksq = 0;
normksq = 0;
for k=1:K
    X_estimated{k} = FacSol.A*diag(FacSol.C(k,:))*FacSol.B{k}';
    TCSksq = TCSksq + norm(Z.original_object{k}(~Z.miss{k})-X_estimated{k}(~Z.miss{k}),'fro')^2;
    normksq = normksq + norm(Z.original_object{k}(~Z.miss{k}),'fro')^2;
end  
TCS = sqrt(TCSksq)/sqrt(normksq); % tensor completion score
%%
figure()
plot(reshape(Z.original_object{k}(~Z.miss{k}),[],1),reshape(X_estimated{k}(~Z.miss{k}),[],1),'*')

