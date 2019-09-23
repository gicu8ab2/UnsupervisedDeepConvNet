
clear

trainZ=hdf5read('./feature_vectors/train_features_cnn_high_compression.h5','train_feature_vecs');
trainY=hdf5read('./feature_vectors/train_labels_cnn_high_compression.h5','train_labels');
trainY=trainY+1;

para.k=100*ones(200,1);
para.cet=200;
para.a=10; para.b=2;
para.maxit=1500; para.num=500;

for ii=1:11,
  ii
  inds=find(trainY==ii);
  spl = MFA_DP(trainZ(:,inds),para);
  save_string=sprintf('save MFA_results_VTCNN_high_compression/MFA_model_params%d.mat spl',ii);
  eval(save_string)
end;

