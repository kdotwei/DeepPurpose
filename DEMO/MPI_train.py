from DeepPurpose.DTI import *
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from time import time

DEBUG = True

"""
Usage:

# Copy your file to all nodes, mind that directory of execution file should be the same
# The purpose of this step is to make sure that all nodes have the same execution file
parallel-scp -A -h <hostfile> <your file's directory> <target directory> 

# Run your file
mpirun -v -np $node --hostfile <your host file> python3 MPI_train.py
"""

def main():
 # Record the start time
 t1 = time()

 # Training Model
 save_path = './model_morgan_cnn_davis'
 drug_encoding = 'Morgan'
 target_encoding = 'CNN'
 learning_rate = 0.001
 batch_size = 256
 train_epoch = 10
 random_seed = 1

 # Load data and generate configuration in process with rank == 0
 X_drug, X_target, y = load_process_DAVIS('./data/', binary=False)

 train, val, test = data_process(X_drug, X_target, y,
         drug_encoding, target_encoding,
         split_method='random', frac=[0.7, 0.1, 0.2], random_seed=random_seed)

 config = generate_config(drug_encoding=drug_encoding,
       target_encoding=target_encoding,
       cls_hidden_dims=[1024, 1024, 512],
       train_epoch=train_epoch,
       LR=learning_rate,
       batch_size=batch_size)
 
 # Train model
 model = model_initialize(**config)
 model.mpi_train(train, val, test)

 # Save the root model
 model.save_model(save_path)

 # Record the end time
 t2 = time()

 print("All models trained successfully.")
 print("Cost about " + str(int(t2-t1)) + " seconds")

if __name__ == '__main__':
 main()

proposal