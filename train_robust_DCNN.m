%script to train Robust DCNN model

addpath('data','label_coding')
path_to_venv_python=sprintf('c:/Users/rtaylor/AppData/Local/Continuum/Anaconda3/envs/tensorflow/python');

disp('Training DCNN on RadioML data')
python_call=sprintf('python CNN_high_compress_train_on_RadioML.py');
system(python_call);

disp('Running feature extraction on training data')
python_call=sprintf('python CNN_high_compress_feat_extract_train.py');
system(python_call);

train_MFA_on_features;  


