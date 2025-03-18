function [G,FacInit,out] = PAR2_AOADMM_EM(Z,options,init)
% Computes the PARAFAC2 model of Z.object with missing entries, which are indicated in Z.miss.
% Applies an Expectation-Maximization (EM) approach in order to estimate the missing entries. 
% The final factors in G are normalized such that factor matrices A and Bk are normalized columnwise and the scaling is moved to factor matrix C


    if ~isfield(options,'iter_start_PAR2Bkconstraint')
        options.iter_start_Bkconstraint = 0;
    end

    FacInit = init;
    G = init;

    % Constraints
    [prox_operators,reg_func] = constraints_to_prox(Z.constrained_modes,Z.constraints,Z.size);
    Z.prox_operators = prox_operators;
    Z.reg_func = reg_func;
    
    R = Z.R;
    K = Z.size{3};
    BtB = cell(K,1);

    Z.normsqr  = 0;
    for k=1:K
        Z.normsqr = Z.normsqr + norm(Z.miss{k}.*Z.object{k},'fro')^2; % precompute squared norm (with zeros at missing entries)
        BtB{k} = G.B{k}'*G.B{k};
    end
    
    XC = cell(K,1);
    YC = cell(K,1);
    LC = cell(K,1);
    rhoC = zeros(K,1);
    YB = cell(K,1);
    XB = cell(K,1);
    LB = cell(K,1);
    rhoB = zeros(K,1);
    
    % update mode A once before the algorithm starts
    XA = zeros(size(G.A));
    YA = zeros(R,R);
    for k=1:K
        XA = XA + Z.object{k}*G.B{k}*diag(G.C(k,:));
        YA = YA + diag(G.C(k,:))*BtB{k}*diag(G.C(k,:));
    end 
    if isfield(Z,'ridge')
        YA = YA + Z.ridge(1)*eye(R,R);
    end
    if Z.constrained_modes(1) % is constrained, use ADMM
        rhoA = trace(YA)/R;
        YA = YA + rhoA/2*eye(R,R);
        LA = chol(YA','lower'); %precompute Cholesky decomposition
        [~] = ADMM_A(XA,LA,rhoA,options);
    else % ALS update
        G.A = XA/YA;
    end

    out.innerIters = zeros(3,1);

    [f_tensors,f_couplings,f_constraints] = PAR2_AOADMM_func_eval();
    f_rel_missing = 0;
    f_total = f_tensors+f_couplings+f_constraints;
    func_val(1) = f_tensors;
    func_coupl(1) = f_couplings;
    func_constr(1) = f_constraints;
    func_rel_missing(1) = f_rel_missing;
    tstart = tic;
    time_at_it(1) = 0;
    %display first iteration
    if strcmp(options.Display,'iter') || strcmp(options.Display,'final')
        fprintf(1,' Iter  f total      f tensors      f couplings    f constraints    f rel missing\n');
        fprintf(1,'------ ------------ -------------  -------------- ---------------- -------------\n');
    end 

    if strcmp(options.Display,'iter')
        fprintf(1,'%6d %12f %12f %12f %12f %12f\n', 0, f_total, f_tensors, f_couplings,f_constraints,f_rel_missing);
    end

    iter = 1;
    
    stop = false;
    while(iter<=options.MaxOuterIters && ~stop)
    
       % precompute A^T*A
       AtA = G.A'*G.A;
       
       % update mode B
       for k=1:K
           XB{k} = Z.object{k}'*G.A*diag(G.C(k,:));
           YB{k} = diag(G.C(k,:))*AtA*diag(G.C(k,:));    
           rhoB(k) = trace(YB{k})/R;
           if isfield(options, 'increase_factor_rhoBk')
               rhoB(k) = options.increase_factor_rhoBk * rhoB(k);
           end
           YB{k} = YB{k} + rhoB(k)/2*eye(R,R); %always coupled 
           if isfield(Z,'ridge')
               YB{k} = YB{k} + Z.ridge(2)*eye(R,R);
           end
           if Z.constrained_modes(2) && iter >= options.iter_start_Bkconstraint
               YB{k} = YB{k} + rhoB(k)/2*eye(size(YB{k}));
           end
           LB{k} = chol(YB{k},'lower'); %precompute Cholesky decomposition 
       end
       [inner_iters] = ADMM_B(XB,LB,rhoB,options);
       out.innerIters(2,iter)= inner_iters;
       for k=1:K
           BtB{k} = G.B{k}'*G.B{k};
       end
       

       
       % update mode C
       for k=1:K   
           XC{k} = diag(G.A'*Z.object{k}*G.B{k});
           YC{k} = AtA.*(BtB{k});
           if isfield(Z,'ridge')
               YC{k} = YC{k} + Z.ridge(3)*eye(R,R);
           end
           if Z.constrained_modes(3) % is constrained, use ADMM (here only precomputations)
               rhoC(k) = trace(YC{k})/R;
               YC{k} = YC{k} + rhoC(k)/2*eye(R,R);
               LC{k} = chol(YC{k}','lower'); %precompute Cholesky decomposition
           else % ALS update (row-wise)
               G.C(k,:) = (YC{k}\XC{k})';
               inner_iters = 1;
           end
       end
       if Z.constrained_modes(3) % is constrained, use ADMM
           [inner_iters] = ADMM_C(XC,LC,rhoC,options);
       end
       out.innerIters(3,iter)= inner_iters;
       
       % update mode A
       XA = zeros(size(G.A));
       YA = zeros(R,R);
       for k=1:K
           XA = XA + Z.object{k}*G.B{k}*diag(G.C(k,:));
           YA = YA + diag(G.C(k,:))*BtB{k}*diag(G.C(k,:));
       end 
       YA_orig = YA;
       if isfield(Z,'ridge')
           YA = YA + Z.ridge(1)*eye(R,R);
       end
       if Z.constrained_modes(1) % is constrained, use ADMM
           rhoA = trace(YA)/R;
           YA = YA + rhoA/2*eye(R,R);
           LA = chol(YA','lower'); %precompute Cholesky decomposition
           [inner_iters] = ADMM_A(XA,LA,rhoA,options);
       else % ALS update
           G.A = XA/YA;
           inner_iters = 1;
       end
       out.innerIters(1,iter)= inner_iters;

       % update missing values
       diff_missing = 0;
       norm_missing = 0;
       for k=1:K 
           Z_object_tempk = G.A*diag(G.C(k,:))*G.B{k}';
           old_missingk = Z.object{k}(~Z.miss{k});
           new_missingk = Z_object_tempk(~Z.miss{k});
           Z.object{k}(~Z.miss{k})= new_missingk;
           diff_missing = diff_missing + norm(old_missingk - new_missingk,2)^2;
           norm_missing = norm_missing + norm(old_missingk,2)^2;
       end 
       if norm_missing > 0
           f_rel_missing = sqrt(diff_missing)/sqrt(norm_missing);
       else
           f_rel_missing = sqrt(diff_missing);
       end
        
       % evaluate function value
       f_tensors_old = f_tensors;
       f_couplings_old = f_couplings;
       f_constraints_old = f_constraints;
       [f_tensors,f_couplings,f_constraints] = PAR2_AOADMM_func_eval();

       f_total = f_tensors+f_couplings+f_constraints;
       func_val(iter+1) = f_tensors;
       func_coupl(iter+1) = f_couplings;
       func_constr(iter+1) = f_constraints;
       func_rel_missing(iter+1) = f_rel_missing;
       time_at_it(iter+1) = toc(tstart);
       stop = evaluate_stopping_conditions_EM(f_tensors,f_couplings,f_constraints,f_tensors_old,f_couplings_old,f_constraints_old,f_rel_missing,options);

        %display
        if strcmp(options.Display,'iter') && mod(iter,options.DisplayIters)==0
            fprintf(1,'%6d %12f %12f %12f %12f %12f\n', iter, f_total, f_tensors, f_couplings,f_constraints,f_rel_missing);
        end
        iter = iter+1;
    end
    
    exit_flag = make_exit_flag(iter,f_tensors,f_couplings,f_constraints,options);
    %save output
    out.f_tensors = f_tensors;
    out.f_couplings = f_couplings;
    out.f_constraints = f_constraints;
    out.exit_flag = exit_flag;
    out.OuterIterations = iter-1;
    out.func_val_conv = func_val;
    out.func_coupl_conv = func_coupl;
    out.func_constr_conv = func_constr;
    out.time_at_it = time_at_it;
    out.func_rel_missing = func_rel_missing;

    % normalize columns of A and B and put norms into C
     for r=1:R
        normAr = norm(G.A(:,r),2);
        if normAr > 0
            G.A(:,r) = G.A(:,r)/normAr;
            G.C(:,r) = G.C(:,r).*normAr;
        end
        for k=1:K
            normBrk = norm(G.B{k}(:,r),2);
            if normBrk > 0
                G.B{k}(:,r) = G.B{k}(:,r)/normBrk;
                G.C(k,r) = G.C(k,r).*normBrk; 
            end
        end
    end

    %display final
    if strcmp(options.Display,'iter') || strcmp(options.Display,'final')
        fprintf(1,'%6d %12f %12f %12f %12f %12f\n', iter-1, f_total, f_tensors, f_couplings,f_constraints,f_rel_missing);
    end
    
    %% nested functions
     function [f_tensors,f_PAR2_couplings,f_constraints] = PAR2_AOADMM_func_eval()
        f_tensors = 0;
        f_PAR2_couplings = 0;
        f_constraints = 0;

        for kk=1:K
            f_tensors = f_tensors + norm(Z.miss{kk}.*(Z.object{kk}-G.A*diag(G.C(kk,:))*G.B{kk}'),'fro')^2;
            f_PAR2_couplings = f_PAR2_couplings + norm(G.B{kk}-G.P{kk}*G.DeltaB,'fro')/norm(G.B{kk},'fro');
            if Z.constrained_modes(2)
                f_constraints = f_constraints + norm(G.B{kk}-G.ZB{kk},'fro')/norm(G.B{kk},'fro');
            end
        end
        f_tensors = f_tensors./Z.normsqr;
        f_PAR2_couplings = f_PAR2_couplings/K;
        if Z.constrained_modes(2)
            f_constraints = f_constraints/K;
        end

        if Z.constrained_modes(1)
            f_constraints = f_constraints + norm(G.A-G.ZA,'fro')/norm(G.A,'fro');
        end
        if Z.constrained_modes(3)
            f_constraints = f_constraints + norm(G.C-G.ZC,'fro')/norm(G.C,'fro');
        end
        if any(Z.constrained_modes)
            f_constraints = f_constraints/sum(Z.constrained_modes);
        end
        
        if isfield(Z,'reg_func')
            if ~isempty(Z.reg_func{1})
                f_tensors = f_tensors + feval(Z.reg_func{1},G.A);
            end
            if ~isempty(Z.reg_func{3})
                f_tensors = f_tensors + feval(Z.reg_func{3},G.C);
            end
            if ~isempty(Z.reg_func{2})
                if strcmp(Z.constraints{2}{1},'tPARAFAC2')
                    f_tensors = f_tensors + feval(Z.reg_func{2},G.B);
                else
                    for kk=1:K
                        f_tensors = f_tensors + feval(Z.reg_func{2},G.B{kk});
                    end
                end
            end
        end
        
        if isfield(Z,'ridge')
            f_tensors = f_tensors + Z.ridge(1)*norm(G.A,'fro')^2 + Z.ridge(3)*norm(G.C,'fro')^2;
            if Z.ridge(2)>0
                 for kk=1:K
                     f_tensors = f_tensors + Z.ridge(2)*norm(G.B{kk},'fro')^2;
                 end
            end
        end
        
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [inner_iter] = ADMM_A(X,L,rho,options)
    %ADMM loop for mode A, where mode A is constrained!
    % changes the global variable G 

        inner_iter = 1;
        rel_primal_res_constr = inf;
        rel_dual_res_constr = inf;
        % ADMM loop
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_constr>options.innerRelDualTol_constr))

            X_inner = X + rho/2*(G.ZA - G.mu_A);

            G.A = (X_inner/L')/L; % forward-backward substitution

            % Update constraint factor (Z_A) and its dual (mu_Z_A)
            oldZ = G.ZA;
            G.ZA = feval(Z.prox_operators{1},(G.A + G.mu_A),rho);
            G.mu_A = G.mu_A + G.A - G.ZA;

            inner_iter = inner_iter + 1; 
            rel_primal_res_constr = norm(G.A - G.ZA,'fro')/norm(G.A,'fro');
            rel_dual_res_constr = norm(oldZ - G.ZA,'fro')/norm(G.mu_A,'fro');
        end
        inner_iter = inner_iter-1;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [inner_iter] = ADMM_C(X,L,rho,options)
    %ADMM loop for mode C, where mode C is constrained!
    % changes the global variable G 

        inner_iter = 1;
        rel_primal_res_constr = inf;
        rel_dual_res_constr = inf;
        % ADMM loop
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_constr>options.innerRelDualTol_constr))
            oldZ = G.ZC;
            for kk=1:K
                X_inner = X{kk} + rho(kk)/2*(G.ZC(kk,:)' - G.mu_C(kk,:)');

                G.C(kk,:) = (L{kk}'\(L{kk}\X_inner))'; % forward-backward substitution
            end
                % Update constraint factor (Z_C) and its dual (mu_Z_C)
                G.ZC = (feval(Z.prox_operators{3},(G.C + G.mu_C),max(rho)));
                G.mu_C = G.mu_C + G.C - G.ZC;

                inner_iter = inner_iter + 1; 
                rel_primal_res_constr = norm(G.C - G.ZC,'fro')/norm(G.C,'fro');
                rel_dual_res_constr = norm(oldZ - G.ZC,'fro')/norm(G.mu_C,'fro');
        end
        inner_iter = inner_iter-1;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [inner_iter] = ADMM_B(X,L,rho,options)
    %ADMM loop for mode A, where mode A is constrained!
    % changes the global variable G 

        inner_iter = 1;
        rel_primal_res_constr = inf;
        rel_dual_res_constr = inf;
        rel_primal_res_coupling = inf;
        rel_dual_res_coupling = inf;
        oldP = cell(K,1);
        % ADMM loop
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_coupling>options.innerRelPrTol_coupl||rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_coupling>options.innerRelDualTol_coupl||rel_dual_res_constr>options.innerRelDualTol_constr))
            rel_primal_res_constr = 0;
            rel_dual_res_constr = 0;
            rel_primal_res_coupling = 0;
            rel_dual_res_coupling = 0;
            for kk=1:K
                X_inner = X{kk} + rho(kk)/2*(G.P{kk}*G.DeltaB - G.mu_DeltaB{kk});
                if Z.constrained_modes(2) && iter >= options.iter_start_Bkconstraint
                    X_inner = X_inner + rho(kk)/2*(G.ZB{kk} - G.mu_B_Z{kk});
                end
                G.B{kk} = (X_inner/L{kk}')/L{kk}; % forward-backward substitution
                % update P_k
                [U,~,V] = svd((G.B{kk} + G.mu_DeltaB{kk})*G.DeltaB','econ');
                oldP{kk} = G.P{kk};
                G.P{kk} = U*V';
            end
            % update DeltaB
            oldDeltaB = G.DeltaB;
            G.DeltaB = zeros(R,R);
            sum_rho_k = 0;
            for kk=1:K
                G.DeltaB = G.DeltaB + rho(kk)*G.P{kk}'*(G.B{kk} + G.mu_DeltaB{kk});
                sum_rho_k = sum_rho_k + rho(kk);
            end
            G.DeltaB = G.DeltaB./sum_rho_k;
            for kk=1:K
                G.mu_DeltaB{kk} = G.mu_DeltaB{kk} + G.B{kk} - G.P{kk}*G.DeltaB;
            end


            % Update constraint factor (Z_B) and its dual (mu_Z_B) if
            % constrained
            if Z.constrained_modes(2) && iter >= options.iter_start_Bkconstraint
                oldZ = G.ZB;

                if strcmp(Z.constraints{2}{1},'tPARAFAC2')
                    G.ZB = feval(Z.prox_operators{2},cellfun(@(x, y) x + y, G.B, G.mu_B_Z, 'UniformOutput', false),rho);
                    for kk=1:K
                        G.mu_B_Z{kk} = G.mu_B_Z{kk} + G.B{kk} - G.ZB{kk};
                        rel_primal_res_constr = rel_primal_res_constr + norm(G.B{kk} - G.ZB{kk},'fro')/norm(G.B{kk},'fro')/K;
                        rel_dual_res_constr = rel_dual_res_constr + norm(oldZ{kk} - G.ZB{kk},'fro')/norm(G.mu_B_Z{kk},'fro')/K;
                    end
                else
                    for kk=1:K
                        G.ZB{kk} = feval(Z.prox_operators{2},(G.B{kk} + G.mu_B_Z{kk}),rho(kk));
                        G.mu_B_Z{kk} = G.mu_B_Z{kk} + G.B{kk} - G.ZB{kk};             
                        % sum up residuals
                        rel_primal_res_constr = rel_primal_res_constr + norm(G.B{kk} - G.ZB{kk},'fro')/norm(G.B{kk},'fro')/K;
                        rel_dual_res_constr = rel_dual_res_constr + norm(oldZ{kk} - G.ZB{kk},'fro')/norm(G.mu_B_Z{kk},'fro')/K;
                    end
                end
            end

            
            for kk=1:K
                rel_primal_res_coupling = rel_primal_res_coupling + norm(G.B{kk} - G.P{kk}*G.DeltaB,'fro')/norm(G.B{kk},'fro')/K;
                rel_dual_res_coupling = rel_dual_res_coupling + norm(oldP{kk}*oldDeltaB - G.P{kk}*G.DeltaB,'fro')/norm(G.mu_DeltaB{kk},'fro')/K;
            end
            inner_iter = inner_iter + 1; 
        end
        inner_iter = inner_iter-1;
    end
end

