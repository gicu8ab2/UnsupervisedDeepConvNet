%script to test weak DCNN model under fgsm attack

addpath('plotting','data','results')
path_to_venv_python=sprintf('c:/Users/rtaylor/AppData/Local/Continuum/Anaconda3/envs/tensorflow/python');

disp('Running feature extraction on batch of adversarial data')
python_call=sprintf('%s CNN_high_compress_under_fgsm_attack.py',path_to_venv_python);
system(python_call); 

disp('Display performance results')
python_call=sprintf('%s plotting/Plot_Classifier_Performance.py',path_to_venv_python);
system(python_call);   


