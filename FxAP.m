function [w, e]= FxAP(ref, d, Nw,order, secpath, mu,eps)
% The arrangement order of Ww is all references corresponding to the 1st secondary source, all references to the 2nd...
% The arrangement order of sec_path is from the 1st secondary source to all error points,..., from the Sth secondary source to all error points
% 
%==========================================================================
    ref_num = size(ref,2);
    err_num = size(d,2);
    spk_num = size(secpath,2)/err_num;
    Nc = size(secpath,1);
    w = zeros(Nw,ref_num*spk_num);
    len = size(d,1);
    e = zeros(len,err_num);
    ref_buffer = zeros(Nw,ref_num);
    con_buffer = zeros(Nc,spk_num);
    e_buffer = zeros(order,err_num);
    Vn = zeros(ref_num*spk_num*Nw,err_num*order);
    filtered_ref = zeros(len,ref_num*spk_num*err_num);
    for i=1:spk_num*err_num
        filtered_ref(:,(i-1)*ref_num+1:i*ref_num) = filter(secpath(:,i),1,ref);
    end
    for n=1:len

        ref_buffer = [ref(n,:);ref_buffer(1:end-1,:)];
        for j=1:spk_num
            con_buffer(:,j) = [sum(w(:,(j-1)*ref_num+1:j*ref_num) .* ref_buffer, 'all'); con_buffer(1:end-1,j)];
        end
        for k=1:err_num
            e(n,k) = d(n,k) + sum(con_buffer.*secpath(:,k:err_num:end),'all');
        end

        e_buffer = [e(n,:);e_buffer(1:end-1,:)];
        if n>order+Nw
            for j=1:err_num
                idx = zeros(spk_num*ref_num,1);
                for i1=1:spk_num
                    idx((i1-1)*ref_num+1:i1*ref_num)=1+(j-1)*ref_num+ref_num*err_num*(i1-1):j*ref_num+ref_num*err_num*(i1-1);
                end
                for i = 1:order
                    Vn(:,(j-1)*order+i) = reshape(filtered_ref(n+1-i:-1:n-Nw+1+1-i,idx),[],1);
                end
            end
            e1_buffer = reshape(e_buffer,[],1);
            VV = Vn.'*Vn;
            update_buffer = -mu*Vn*((VV+eps*eye(err_num*order))\e1_buffer);
            update_buffer = reshape(update_buffer,Nw,ref_num*spk_num);
            w = w+update_buffer;
        end
    end

end



