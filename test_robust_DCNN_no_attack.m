%script to test Robust DCNN model with natural inputs (no adversarial attack)

addpath('plotting','data','results')
path_to_venv_python=sprintf('c:/Users/rtaylor/AppData/Local/Continuum/Anaconda3/envs/tensorflow/python');

disp('Running feature extraction on batch of test data')
python_call=sprintf('%s CNN_high_compress_feat_extract_test.py',path_to_venv_python);
system(python_call); 

disp('Maximum likelihood classification from p(z|c) on test batch')
evaluate_p_z_given_c_test_batch;

disp('Plotting performance')
plot_performance_by_snr_no_attack

