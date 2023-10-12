function [exit_flag] = make_exit_flag(iter,f_tensors,f_PAR2_couplings,f_constraints,options)
% which condition caused stop?

if iter>options.MaxOuterIters
    exit_flag = 'maxIterations';
else
    if f_tensors < options.AbsFuncTol 
        exit_flag.f_tensors = "AbsFuncTol";
    else
        exit_flag.f_tensors = "RelFuncTol";
    end
    if f_PAR2_couplings < options.AbsFuncTol 
        exit_flag.f_couplings = "AbsFuncTol";
    else
        exit_flag.f_couplings = "RelFuncTol";
    end
    if f_constraints < options.AbsFuncTol 
        exit_flag.f_constraints = "AbsFuncTol";
    else
        exit_flag.f_constraints = "RelFuncTol";
    end
end
   
end

