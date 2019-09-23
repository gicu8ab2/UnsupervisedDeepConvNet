function spl = MFA_DP(X0,para)
k = para.k; cet = para.cet; flag = 0;
maxit = para.maxit; num = para.num;
[p,n] = size(X0); T = length(k); len = zeros(T,1);
Xm = mean(X0,2); X = X0 - Xm*ones(1,n);
H = ones(T,n)/T; G = H; 
gamma0 = 10^0; eta0 = 0;
beta0 = 10^0; alpha0 = p;
c0 = 10^-6; d0 = 10^-6;
g0 = 10^-6; h0 =10^-6;
U = [betarnd(ones(T-1,1),cet*ones(T-1,1)); 1];

X2 = X.*X;  ord = randperm(n);
z = cell(T,1); w = cell(T,1);
pai = cell(T,1); 
e0 = zeros(T,1); f0 = zeros(T,1);
A = cell(T,1); S = cell(T,1);
mu = cell(T,1); Phi = cell(T,1);
C = cell(T,1); AA = cell(T,1); 
AX = cell(T,1); Amu = cell(T,1);
SS = cell(T,1); 
L = cell(T,1); Su = cell(T,1);
minlen = 10; u = zeros(n,1);

for t = 1:T
    z{t} = ones(k(t),1); w{t} = 0*ones(k(t),1);
    pai{t} = 1/k(t)*ones(k(t),1);
    e0(t) = para.a/k(t); f0(t) = para.b*(1 - 1/k(t));
    A{t} = randn(p,k(t)); S{t} = zeros(k(t),n);
    mu{t} = X(:,ord(t)); 
    Phi{t} = 10^0*ones(p,1);
end

spl.A = cell(T,1); spl.z = cell(T,1); spl.w = cell(T,1);
spl.mu = cell(T,1); spl.Phi = cell(T,1); spl.pai = cell(T,1);
spl.H = zeros(T,n); spl.qai = zeros(T,1);  spl.U = zeros(T,1);
spl.S1 = cell(T,1); spl.S2 = cell(T,1); spl.X_hat = zeros(p,n);
for t = 1:T
    spl.A{t} = zeros(p,k(t)); spl.z{t} = zeros(k(t),1);
    spl.w{t} = zeros(k(t),1); spl.mu{t} = zeros(p,1);
    spl.Phi{t} = zeros(p,1); spl.pai{t} = zeros(k(t),1);
    spl.S1{t} = zeros(k(t),1); spl.S2{t} = zeros(k(t),k(t));
end

iter = 0; 
while (iter<maxit) 
    iter  = iter + 1;
     for t = 1:T
        C{t} = A{t}'.*repmat(Phi{t}',[k(t),1]);
        AA{t} = C{t}*A{t}; AX{t} = C{t}*X; Amu{t} = C{t}*mu{t};

        zw = z{t}.*w{t};
        Yt = diag(zw)*(AX{t}-Amu{t}*ones(1,n));
        L{t} = chol((zw*zw').*AA{t}+beta0*eye(k(t)));
        Su{t} = L{t}\(L{t}'\Yt);
        
        G(t,:) = sum(log(1-U(1:t-1)+realmin)) + log(U(t)+realmin)  - p*log(2*pi)/2 + 0.5*(-2*sum(log(diag(L{t})))) ...
                    + 0.5*k(t)*log(beta0) + 0.5*sum(log(Phi{t})) - 0.5*(Phi{t}'*X2 ...
                    + Phi{t}'*(mu{t}.^2) - 2*(mu{t}'.*Phi{t}')*X) + 0.5*sum(Yt.*Su{t});
    end  
    H = exp(G + repmat(-max(G,[],1),[T,1])) + realmin;
    H = H./repmat(sum(H,1),[T 1]);
    
    for i = 1:n, u(i) = randsample(T,1,true,H(:,i));  end
    spl.J(iter) = 0; for i = 1:n, spl.J(iter) = spl.J(iter) + G(u(i),i); end
    g = g0 + T-1;
    h = h0 - sum(log(1-U(1:T-1)+realmin));
    cet = gamrnd(g,1/h);

    for t = 1:T-1
        U(t) = betarnd(1+length(find(u==t)),cet+length(find(u>t)));
    end
    U(T) = 1; qai = cumprod([1;1-U(1:T-1)]).*U;
    
    X_hat = zeros(size(X)); res = zeros(size(X));
    for t = 1:T
        ndx = find(u==t); len(t) = length(ndx);
        if len(t) > minlen        
 
        S{t}(:,ndx) = Su{t}(:,ndx) + (L{t}\randn(k(t),len(t)));
        SS{t} = S{t}(:,ndx)*S{t}(:,ndx)';    
            
        for m = 1:k(t)
            z{t}(m) = 0; w{t}(m) = 0;
            tmp3 = AA{t}(m,m)*SS{t}(m,m);
            tmp4 = AX{t}(m,ndx)*S{t}(m,ndx)'-AA{t}(m,:)*(z{t}.*w{t}.*SS{t}(:,m))...
                                      -Amu{t}(m)*ones(1,len(t))*S{t}(m,ndx)';
            tmpb = (gamma0^(-1) + tmp3)^(-1);
            tmpa = tmpb*(gamma0^(-1)*eta0 + tmp4);
            tmprr = (log(pai{t}(m)+eps) + 1/2*log(tmpb) + 1/2*(tmpb^-1)*(tmpa^2)) ...
                         - (log(1-pai{t}(m)+eps) + 1/2*log(gamma0) + 1/2*(gamma0^-1)*(eta0^2));
            if  rand <  1/(1+exp(-tmprr)) 
                z{t}(m) = 1; w{t}(m) = normrnd(tmpa,sqrt(tmpb));
            else
                z{t}(m) = 0; w{t}(m) = normrnd(eta0,sqrt(gamma0));
            end
        end 
        pai{t} = betarnd(e0(t) + z{t}, f0(t) + 1 - z{t});
        
        mum = mean(X(:,ndx)-A{t}*diag(z{t}.*w{t})*S{t}(:,ndx),2);
        tmp1 = 1./(Phi{t}*len(t));
        tmp2 = tmp1.*(Phi{t}.*mum*len(t));
        mu{t} = normrnd(tmp2,(tmp1).^(0.5));
        
        B = [diag(z{t}.*w{t})*S{t}(:,ndx)];
        B2 = B*B' + 10^-6*eye(k(t)); 
        [V,D] = eig(B2); 
        tmp0 = alpha0*ones(k(t),p) + diag(D)*Phi{t}';
        tmp1 = (ones(k(t),1)*Phi{t}')./tmp0;
        tmp2 = tmp0.^(-0.5);
        Tmp = V*(tmp1.*(V'*B*(X(:,ndx)-mu{t}*ones(1,len(t)))') + tmp2.*randn(k(t),p));
        A{t} = Tmp';
        
        X_hat(:,ndx) = A{t}*diag(z{t}.*w{t})*S{t}(:,ndx) + mu{t}*ones(1,len(t));    
        
        res(:,ndx) = X(:,ndx) - X_hat(:,ndx);
        if flag
            Phi{t} = min(gamrnd(c0+1/2*len(t), 1./(d0+1/2*sum( res(:,ndx).^2,2 ))), 1E6);
        else
            tmp = min(gamrnd(c0+0.5*len(t)*p,1/(d0+0.5*sum(sum(res(:,ndx).^2)))),1E6);
            Phi{t} = tmp*ones(p,1);
        end

        else % No data in this cluster, draw from prior
        pai{t} = betarnd(e0(t)*ones(k(t),1), f0(t)*ones(k(t),1));
        z{t} = binornd(1*ones(k(t),1),pai{t});
        w{t} = normrnd(eta0*ones(k(t),1),sqrt(gamma0)*ones(k(t),1));
        Phi{t} = 1E6*ones(p,1); A{t} = zeros(p,k(t)); mu{t} = zeros(p,1);
        end
    end
   
    err(iter) = sqrt(sum(sum(res.^2,2))/p/n);
    
    if iter > maxit - num
        for t = 1:T
            spl.A{t} = spl.A{t} + A{t}/num;
            spl.z{t} = spl.z{t} + z{t}/num;
            spl.w{t} = spl.w{t} + w{t}/num;
            spl.pai{t} = spl.pai{t} + pai{t}/num;
            spl.mu{t} = spl.mu{t} + (mu{t}+Xm)/num;
            spl.Phi{t} = spl.Phi{t} + Phi{t}/num;
            ndx = find(u==t); len(t) = length(ndx);
            if len(t) > minlen
                spl.S1{t} = spl.S1{t} + mean(S{t}(:,ndx),2)/num;
                spl.S2{t} = spl.S2{t} + SS{t}/len(t)/num;
            end
        end
        spl.H = spl.H + H/num;
        spl.U = spl.U + U/num;
        spl.qai = spl.qai + qai/num;
        spl.X_hat = spl.X_hat + (X_hat+Xm*ones(1,n))/num;
    end

%     if mod(iter,50)==0
%     figure(1); 
%     [vv,nn] = sort(-qai);
%     subplot(3,3,1); imagesc(res); colorbar; 
%     title(['Residue '  num2str(num2str(err(iter))) ' Iteration ' num2str(iter)]);
%     subplot(3,3,2); imagesc(H); colorbar; title('Occupation');
%     subplot(3,3,3); bar(1:T,qai); title('\pi');
%     for t = 1:min(T,6)
%         subplot(3,3,3+t); plot(z{nn(t)}.*w{nn(t)},'r+'); 
%         title(['z.*w for cluster ' num2str(nn(t)) '  Num: ' num2str(length(find(z{nn(t)}==1)))]);
%     end
%     drawnow;
%     end
    disp(['Iteration ' num2str(iter) ' Residue ' num2str(num2str(err(iter)))]);
end
return