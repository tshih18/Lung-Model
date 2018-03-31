###################################################
#
#   Script to execute the prediction
#
##################################################

import os, sys
import ConfigParser


# -------- Load settings from Config file -------------------------------------
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))

#name of the experiment!!
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')
# -----------------------------------------------------------------------------

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

# Create a folder for the results if not existing already
result_dir = name_experiment
print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(result_dir):
	pass
elif sys.platform=='win32':
	os.system('md ' + result_dir)
else:
	os.system('mkdir -p ' + result_dir)


# Run the testing script
if nohup:
	print "\n2. Run the prediction on GPU  with nohup"
	os.system(run_GPU +' nohup python -u ./src/predict_lungCT.py > ' +'./'+name_experiment+'/'+name_experiment+'_prediction.nohup')
else:
	print "\n2. Run the prediction on GPU (no nohup)"
	os.system(run_GPU +' python ./src/predict_lungCT.py')
