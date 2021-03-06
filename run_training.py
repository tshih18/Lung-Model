###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
import configparser

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['KERAS_BACKEBND'] = 'tensorflow'


# -------- Load settings from Config file -------------------------------------
config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))

#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')
# -----------------------------------------------------------------------------

run_GPU = '' # if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

# Create a folder for the results
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
	print("Dir already existing")
elif sys.platform=='win32':
	os.system('mkdir ' + result_dir)
else:
	os.system('mkdir -p ' +result_dir)

print("copy the configuration file in the results folder")
if sys.platform=='win32':
	os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
	os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

# Run the training script
if nohup:
	print("\n2. Run the training on GPU with nohup")
	os.system(run_GPU +' nohup python -u ./src/train_lungCT.py > ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
else:
	print("\n2. Run the training on GPU (no nohup)")
	os.system(run_GPU +' python ./src/train_lungCT.py')
