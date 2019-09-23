function [A,mu,dd,lambda,a]=extract_relevant_params(spl)

%cluster_inds=find(spl.qai>.005);  
cluster_inds=find(spl.qai>.001);  
%cluster_inds=find(spl.qai>.01);  
lambda=spl.qai(cluster_inds);  lambda=lambda/sum(lambda);
K=length(cluster_inds);
for k=1:K,dd(k)=sum(spl.pai{cluster_inds(k)}>.3);end;
%for k=1:K,dd(k)=sum(spl.pai{cluster_inds(k)}>.4);end;
d=max(dd);

[A1,mu1]=get_posterior_mfa(spl); 
for k=1:K,
  jj=cluster_inds(k);
  a(k)=spl.Phi{jj}(1);
  [val,inds]=sort(spl.pai{jj},'descend');  
  selected_bases_idx=inds(1:d);
  A(:,:,k)=A1{jj}(:,selected_bases_idx);  
  mu(:,k)=mu1{jj};
end;

