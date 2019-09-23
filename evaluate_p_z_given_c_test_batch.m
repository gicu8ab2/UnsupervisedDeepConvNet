%Script file to generate pdf from BNP-MFA
clear

testZ=hdf5read('./feature_vectors/test_features_cnn_high_compression.h5','test_feature_vecs');
testY=hdf5read('./feature_vectors/test_labels_resnet34_50_epochs.h5','test_labels');
testY=testY+1;
filename='p_z_given_c_test.h5';
  
num_classes = 11;
num_pts = size(testZ,2);
p_z_given_c=zeros(num_classes,num_pts);
for c=1:num_classes,
  c
  load_string=sprintf('load MFA_results_VTCNN_high_compression/MFA_model_params%d.mat spl',c);
  eval(load_string);
  [A,mu,dd,lambda,a]=extract_relevant_params(spl);
  [N,d,K]=size(A);
  tmp=zeros(K,num_pts);
  for k=1:K,
    Z_minus_mu=bsxfun(@minus,testZ,mu(:,k));
    A_k=A(:,1:dd(k),k);
    a(k)=1;
    Omega_k_inv = eye(N)-A_k*inv((1/a(k))*eye(dd(k))+A_k'*A_k)*A_k';
    L_k=chol(Omega_k_inv);
    p_z_given_c(c,:)=p_z_given_c(c,:)+exp(log(lambda(k))-.5*sum((L_k*Z_minus_mu).^2)); 
  end
end;
hdf5write(filename,'p_z_given_c',p_z_given_c);

[~,c_inds]=max(p_z_given_c,[],1);
fprintf('Test accuracy is: %2.2d\n',sum(c_inds==testY')/length(c_inds))



