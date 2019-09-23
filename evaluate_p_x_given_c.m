%Script file to generate soft decision labels from BNP-MFA
clear

testX=hdf5read('data/RadioML_training_data.h5','training_data');
testY=hdf5read('data/RadioML_training_labels.h5','training_labels');
testY=testY+1;

num_classes = 11;
num_pts = size(testX,2);
label=[];
for c=1:num_classes,
  c
  load_string=sprintf('load BNPMFA_Results_inputs/MFA_model_params%d.mat spl',c);
  eval(load_string);
  [A,mu,dd,lambda,a]=extract_relevant_params(spl);
  [N,d,K]=size(A);
  tmp=zeros(K,num_pts);
  for k=1:K,
    [k,K]
    X_minus_mu=bsxfun(@minus,testX,mu(:,k));
    A_k=A(:,1:dd(k),k);
    a(k)=1;
    Omega_k_inv = eye(N)-A_k*inv((1/a(k))*eye(dd(k))+A_k'*A_k)*A_k';
    L_k=chol(Omega_k_inv);
    tmp(k,:)=exp(log(lambda(k))-.5*sum((L_k*X_minus_mu).^2)); 
  end
  label=[label;tmp];
  num_clusters_vec(c)=K;
end;


hdf5write('soft_labels_RadioML_train.h5','soft_label',label);
hdf5write('num_clusters_vec.h5','num_clusters_vec',num_clusters_vec);



