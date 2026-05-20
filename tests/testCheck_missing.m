function tests = testCheck_missing
tests = functiontests(localfunctions);
end

function testMissingMaskNotCellError(testCase)
%verify that function actually throws an error for a missing mask that is
%not a cell
    sz_A = 20; %I
    sz_C = 20; %K
    sz_B = 20*ones(1,sz_C); %J_k
    Zsize = {sz_A,sz_B,sz_C};
    verifyError(testCase, @()check_missing(rand(4,4),Zsize),'PAR2:missingData:MissingMaskNotCell')
end

function testMissingMaskWrongLengthError(testCase)
%verify that function actually throws an error for a missing mask that is a
%cell but has the wrong length
sz_A = 20; %I
sz_C = 20; %K
sz_B = 20*ones(1,sz_C); %J_k
Zsize = {sz_A,sz_B,sz_C};
    for k = 1:30
        n_rows = 20;
        n_cols = 20;
        n_k    = n_rows * n_cols;
        mask_k = true(n_rows, n_cols);
        mask_k(randperm(n_k, round(0.5 * n_k))) = false;
        miss_cell_PAR2{k} = mask_k;
    end
    verifyError(testCase, @()check_missing(miss_cell_PAR2,Zsize),'PAR2:missingData:MissingMaskNotCell')
end

function testMissingMaskConvertToLogical(testCase)
%verify that function converts a binary mask into a logical mask correctly
sz_A = 20; %I
sz_C = 20; %K
sz_B = 20*ones(1,sz_C); %J_k
Zsize = {sz_A,sz_B,sz_C};
    for k = 1:sz_C
        n_rows = 20;
        n_cols = 20;
        n_k    = n_rows * n_cols;
        mask_k = true(n_rows, n_cols);
        mask_k(randperm(n_k, round(0.5 * n_k))) = false;
        mask_k_binary = double(mask_k);
        miss_cell_binary{k} = mask_k_binary;
        miss_cell_logical{k} = mask_k;
    end
    verifyEqual(testCase, check_missing(miss_cell_binary,Zsize),miss_cell_logical)
end

function testMissingMaskSliceNotLogical(testCase)
%verify that function throws an error for a missing mask that is not
%logical or binary
sz_A = 20; %I
sz_C = 20; %K
sz_B = 20*ones(1,sz_C); %J_k
Zsize = {sz_A,sz_B,sz_C};
    for k = 1:sz_C
        n_rows = 20;
        n_cols = 20;
        n_k    = n_rows * n_cols;
        mask_k = true(n_rows, n_cols);
        mask_k(randperm(n_k, round(0.5 * n_k))) = false;
        miss_cell{k} = double(mask_k);
    end
    miss_cell{2}(1,1) = 3;
    verifyError(testCase, @()check_missing(miss_cell,Zsize),'PAR2:missingData:MissingMaskSliceNotLogical')
end

function testPAR2maskSliceSizeMismatch(testCase)
%verify that function throws an error for a missing mask does not have the
%correct dimensions (I or Jk)
sz_A = 21; %I
sz_C = 20; %K
sz_B = 20*ones(1,sz_C); %J_k
Zsize = {sz_A,sz_B,sz_C};
    for k = 1:sz_C
        n_rows = 20;
        n_cols = 20;
        n_k    = n_rows * n_cols;
        mask_k = true(n_rows, n_cols);
        mask_k(randperm(n_k, round(0.5 * n_k))) = false;
        miss_cell{k} = mask_k;
    end
    verifyError(testCase, @()check_missing(miss_cell,Zsize),'PAR2:missingData:PAR2maskSliceSizeMismatch')
end


