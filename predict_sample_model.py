#!/usr/bin/env python3

import os
import pickle
from collections import namedtuple

from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb


#######################################################################

OPS = ['max_pool_3x3',
				'avg_pool_3x3',
				'skip_connect',
				'sep_conv_3x3',
				'sep_conv_5x5',
				'dil_conv_3x3',
				'dil_conv_5x5'
				]

def get_genotype_from_arch(arch):
	normal = []
	reduce = []
	sample_normal = arch[0]
	for nodes, op in sample_normal:
		op_name = OPS[op]
		normal.append((op_name, nodes))
		
	sample_reduce = arch[1]
	for nodes, op in sample_reduce:
		op_name = OPS[op]
		reduce.append((op_name, nodes))
		
	sample_darts = Genotype(normal=normal, normal_concat=[2, 3, 4, 5], reduce=reduce, reduce_concat=[2, 3, 4, 5])
	return sample_darts

#######################################################################


# Default dirs for models
# Note: Uses 0.9 as the default models, switch to 1.0 to use 1.0 models
version = '0.9'

current_dir = os.path.dirname(os.path.abspath(__file__))
models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
model_paths_0_9 = {
	model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
	for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
model_paths_1_0 = {
	model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
	for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

# If the models are not available at the paths, automatically download
# the models
# Note: If you would like to provide your own model locations, comment this out
if not all(os.path.exists(model) for model in model_paths.values()):
	nb.download_models(version=version, delete_zip=True,
						download_dir=current_dir)
	
# Load the performance surrogate model
#NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
#NOTE: Defaults to using the default model download path
print("==> Loading performance surrogate model...")
ensemble_dir_performance = model_paths['xgb']
print(ensemble_dir_performance)
performance_model = nb.load_ensemble(ensemble_dir_performance)

# Load the runtime surrogate model
#NOTE: Defaults to using the default model download path
print("==> Loading runtime surrogate model...")
ensemble_dir_runtime = model_paths['lgb_runtime']
runtime_model = nb.load_ensemble(ensemble_dir_runtime)

# Option 1: Create a DARTS genotype
print("==> Creating test configs...")
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

print("		==> Read sampled model archs in Darts space...")
model_arch = pickle.load(open("../maestro/tools/frontend/nasbench/model_arch.pickle", "rb"))

print("		==> Convert each models arch to Genotype and predict performance...")
index = 0
accuracy = []
latency = []
for arch in model_arch:
	genotype_config = get_genotype_from_arch(arch)

	# Predict
	#print("==> Predict runtime and performance...")
	prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=True)
	
	runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")
	
	print(index, "Genotype architecture performance: %f, runtime %f" %(prediction_genotype, runtime_genotype))
	accuracy.append(prediction_genotype)
	latency.append(runtime_genotype)
	
	index += 1
	
pickle.dump(accuracy, open("sample_accuracy.pickle", "wb"))
pickle.dump(latency, open("sample_latency.pickle", "wb"))	