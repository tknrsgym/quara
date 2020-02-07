function [output1,output2] = uniqueValueList_idList_from_nonuniqueValueList(list_val, eps)
%UNIQUEVALUELIST_IDLIST_FROM_NONUNIQUEVALUELIST returns a list of unique
%values in list_val and a list of equivalent ids.
% - eps : tolerance parameter. If two values satisfy |v1 - v2| < eps, we identify v1 and v2. 
%   
    assert(eps >= 0)
    
    num_val = size(list_val, 1);
    list_checked = [];
    id_val_unique = 0;
    for i_val = 1:num_val 
        isChecked = false;
        num_checked = size(list_checked, 2);
        for i_checked = 1:num_checked
            if list_checked(i_checked) == i_val
                isChecked = true;
                break;
            end
        end

        if isChecked == true
            continue;
        end

        val_i = list_val(i_val);    
        id_val_unique = id_val_unique + 1;
        val_unique(id_val_unique) = val_i;
        list_i_val = [i_val];
        list_checked = [list_checked, i_val];    
        for j_val = i_val +1:num_val
            val_j = list_val(j_val);
            dif = abs(val_i - val_j);
            if(abs(dif) < eps)
                list_i_val = [list_i_val, j_val];
                list_checked = [list_checked, j_val];
            end
        end
        list_id(id_val_unique).vec = list_i_val;
    end
    
    output1 = val_unique;
    output2 = list_id;
end

