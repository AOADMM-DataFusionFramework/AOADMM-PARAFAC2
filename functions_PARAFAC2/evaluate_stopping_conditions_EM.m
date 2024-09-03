function [stop] = evaluate_stopping_conditions_EM(f_tensors,f_PAR2_couplings,f_constraints,f_tensors_old,f_PAR2_couplings_old,f_constraints_old,f_rel_missing,options)
% wether or not to stop
stop_tensors = false;
stop_PAR2_couplings = false;
stop_constraints = false;
stop_missing = false;

if f_tensors_old>0
    f_tensors_rel_change = abs(f_tensors_old-f_tensors)/f_tensors_old;
else
    f_tensors_rel_change = abs(f_tensors_old-f_tensors);
end
if f_tensors < options.AbsFuncTol || f_tensors_rel_change < options.OuterRelTol
    stop_tensors = true;
end

if f_PAR2_couplings_old>0
    f_couplings_rel_change = abs(f_PAR2_couplings_old-f_PAR2_couplings)/f_PAR2_couplings_old;
else
    f_couplings_rel_change = abs(f_PAR2_couplings_old-f_PAR2_couplings);
end
if f_PAR2_couplings < options.AbsFuncTol || f_couplings_rel_change < options.OuterRelTol
    stop_PAR2_couplings = true;
end

if f_constraints_old>0
    f_constraints_rel_change = abs(f_constraints_old-f_constraints)/f_constraints_old;
else
    f_constraints_rel_change = abs(f_constraints_old-f_constraints);
end
if f_constraints < options.AbsFuncTol || f_constraints_rel_change < options.OuterRelTol
    stop_constraints = true;
end

if f_rel_missing < options.OuterRelTol
    stop_missing = true;
end

stop = stop_tensors & stop_PAR2_couplings & stop_constraints & stop_missing;


end

