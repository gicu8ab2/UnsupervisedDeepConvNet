%set params
num_classes=11;

test_SNRs=hdf5read('test_SNRs_array.h5','test_SNRs_array');
p_z_given_c_test_adv=hdf5read('./results/p_z_given_c_test_adv.h5','p_z_given_c');
testY=hdf5read('./feature_vectors/test_labels_resnet34_500_epochs.h5','test_labels');
testY=testY'+1;
ii=0;
acc=zeros(num_classes,1);
for snr=-20:2:18,
  ii=ii+1;
  inds=find(test_SNRs==snr);
  [~,c_inds]=max(p_z_given_c_test_adv(:,inds),[],1);
  acc(ii)=sum(c_inds==testY(inds))/length(inds);
end;

gcf=figure,set(gcf,'color',[1,1,1])
plot(-20:2:18,acc,'linewidth',5)
grid
title('P_{cc} vs SNR','fontsize',25)
xlabel('SNR (dB)','fontsize',16)
ylabel('Probability of correct classification','fontsize',16)


