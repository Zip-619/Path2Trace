# Standard Library
import os

# Third Party
import torch

# Local Modules
from .autoencoders import AE

###############
# GPU Setting #
###############
os.environ["CUDA_VISIBLE_DEVICES"]="0"   # comment this line if you want to use all of your GPUs
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


####################
# Data preparation #
####################
def prepare_dataset(sequential_data) :
    data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
    seq_len = data_in_tensor.shape[1]
    return data_in_tensor, seq_len


##################################################
# QuickEncode : Encoding & Decoding & Final_loss #
##################################################
def QuickEncode(input_data, 
                embedding_dim=32, 
                learning_rate = 1e-3, 
                every_epoch_print = 100, 
                epochs = 10000, 
                patience = 20, 
                max_grad_norm = 0.005):
    refined_input_data, seq_len = prepare_dataset(input_data)
    model = AE(in_dim=seq_len, hidden_dim1=200,hidden_dim2=156,\
        learning_rate= learning_rate, out_dim=embedding_dim, \
            every_epoch_print= every_epoch_print, \
                epochs = epochs, patience= patience,max_grad_norm= max_grad_norm)
    final_loss = model.fit(refined_input_data)
    
    # recording_results
    embedded_points = model.encode(refined_input_data)
    decoded_points = model.decode(embedded_points)

    return embedded_points.cpu().data, decoded_points.cpu().data, final_loss

