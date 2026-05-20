function [Zmiss_struct] = check_missing(Zmiss_struct,Zsize)
% checks if the missing value mask has
% correct dimensions and is logical. Binary masks are converted to logical
% ones. 
    
    K = Zsize{3};
    if ~iscell(Zmiss_struct) || length(Zmiss_struct) ~= K
        error('PAR2:missingData:MissingMaskNotCell', ...
            'Z.miss must be a cell array of length %d.', K);
    end
    for k = 1:K
        if ~islogical(Zmiss_struct{k})
            if isnumeric(Zmiss_struct{k}) && all(ismember(Zmiss_struct{k}(:), [0 1]))
                Zmiss_struct{k} = logical(Zmiss_struct{k});
            else
                error('PAR2:missingData:MissingMaskSliceNotLogical', ...
                    'Z.miss{%d} must be a logical or binary (0/1) array.', k);
            end
        end
        if ~isequal([Zsize{1},Zsize{2}(k)], size(Zmiss_struct{k}))
            error('PAR2:missingData:PAR2maskSliceSizeMismatch', ...
                'Z.miss{%d} size does not match Z.size{%d}.', k, k);
        end
    end

end

