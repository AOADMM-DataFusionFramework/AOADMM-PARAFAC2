%% script_test_AOADMM tPARAFAC2 
close all
clear all
%% add AO-ADMM solver functions to path
addpath(genpath('.../prox_operators/matlab/'))
addpath(genpath('./functions_PARAFAC2'))
addpath(genpath('./data_for_test_scripts'))
addpath(genpath('.../tensor_toolbox-v3.2.1'))

%% plot configurations
set(0,'DefaultAxesFontSize',22)
set(0,'DefaultLineLineWidth',4)
set(0,'DefaultTextFontSize',22)
set(0,'DefaultLineMarkerSize',2) %Marker size
%% load PARAFAC2 data 

A = load("gnd_factors.mat", "A").A;
B_double = load("gnd_factors.mat", "B").B;
C = load("gnd_factors.mat", "C").C;
K = size(C,1);

% Initialize cell array
B = cell(1,size(B_double, 1));

% Loop through the first dimension and assign 2D slices to cells
for k = 1:size(B, 2)
    B{k} = squeeze(B_double(k, :, :));
end

noisy_data = load("noisy_dataset.mat", "dataset");

sz_A = size(A, 1);
sz_C = size(C, 1);
sz_B = 80*ones(1,sz_C);
R = 3;
K = sz_C;

% create Z.object 

for k=1:K
    Z.object{k} = noisy_data.dataset(:,:,k);
end

%% Parameters for AOADMM
constrained_modes = [0 1 1]; % for modes [A,B,C]: 1 if the mode is constrained in some way, 0 otherwise!

constraints = cell(length(constrained_modes),1); % cell array of length number of modes containing the type of constraint or regularization for each mode, empty if no constraint
%specify constraints-regularizations for each mode, find the options in the file "List of constraints and regularizations.txt"
constraints{2} = {'tPARAFAC2',1000}; % temporal smoothnesss penalty, 1: is the temporal smoothness strength
constraints{3} = {'non-negativity'};

%% add optional ridge regularization performed via primal variable updates, not proximal operators (for no ridge leave field empty), will automatically be added to function value computation
Z.ridge = [100,0,100]; % penalties for modes [A,Bk,C]
%%
Z.size  = {sz_A,sz_B,sz_C};
Z.constrained_modes = constrained_modes;
Z.constraints = constraints;
Z.R = R;
%% Create random initialization
[prox_operators,reg_func] = constraints_to_prox(Z.constrained_modes,Z.constraints,Z.size);

init.A = rand(sz_A,R);
init.C = rand(sz_C,R);

init.DeltaB = rand(R,R);
for k=1:K
    init.mu_DeltaB{k} = rand(sz_B(k),R);
end

for k=1:K
    init.B{k} = rand(sz_B(k),R);
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
        % init.ZB{k} = feval(prox_operators{2},init.ZB{k},1);
        init.mu_B_Z{k} = rand(sz_B(k),R);
    end
end
%% Fit using AOADMM  

options.Display ='iter'; %  set to 'iter' or 'final' or 'no'
options.DisplayIters = 100;
options.MaxOuterIters = 6000;
options.MaxInnerIters = 5;
options.AbsFuncTol   = 1e-14;
options.OuterRelTol = 1e-8;
options.innerRelPrTol_coupl = 1e-4;
options.innerRelPrTol_constr = 1e-4;
options.innerRelDualTol_coupl = 1e-4;
options.innerRelDualTol_constr = 1e-4;
%% set additional (optional) options for diificult constraints/regularizations on mode Bk of PARAFAC2
%options.iter_start_Bkconstraint = 100; % set the number of iterations after which the constraint on Bk will be active 
%options.increase_factor_rhoBk = 10; % set the factor by which the automatically selected value of rho_Bk is increased
%% fit the model

fprintf('AOADMM PARAFAC2 \n')
tic
[FacSol,FacInit,out] = PAR2_AOADMM(Z,options,init); 
toc

%% FMS, normalization and permutation
% in normSol, factor matrices A and Bk are normalized columnwise, the
% scaling is moved to factor matrix C, all factors are permute to best
% match the original factors
[FMS_A,normSol.A,FLAG_A,PERM_A] = score(ktensor(ones(R,1),FacSol.A),ktensor(ones(R,1),A),'lambda_penalty',false);
normSol.A = normSol.A.U{1};

largeB = [];
SollargeB = [];
prod=1;
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

disp(["FMS A",FMS_A,"FMS B",FMS_B,"FMS C",FMS_C,"TOTAL",FMS_A*FMS_B*FMS_C]);


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


