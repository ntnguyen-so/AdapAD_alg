## This is a config file for AdapAD ##
# The hyper-parameters are set such that AdapAD achieves the highest precisions #

# Note: comment/uncomment one of the following data sources to test
data_source_path = "./01_data/01_label/Wave_height.csv"
# data_source_path = "./01_data/01_label/Tide_pressure.validation_stage.csv"
# data_source_path = "./01_data/01_label/Seawater_temperature.csv"

# Note: comment/uncomment one of the following data sources to test.
# This should align with data_source_path
data_source = "Wave_height"
# data_source = "Tide_pressure"
# data_source = "Seawater_temperature"


epoch_train = 3000
lr_train = 0.00005
epoch_update = 100
lr_update = 0.00005
update_G_epoch = 100
update_G_lr = 0.00005
LSTM_size = 100
LSTM_size_layer = 3

def init_config():    
    global update_G_epoch 
    global update_G_lr

    predictor_config = dict()
    predictor_config['lookback_len'] = 3
    predictor_config['prediction_len'] = 1
    predictor_config['train_size'] = 5*predictor_config['lookback_len'] + \
                                     predictor_config['prediction_len'] 
    
    minimal_threshold = None
    
    if 'Tide_pressure' == data_source:
        value_range_config = (700, 770)
        minimal_threshold = 0.0036
        
        update_G_epoch = 5
        update_G_lr = 0.00005        
    elif "Wave_height" == data_source:
        value_range_config = (0, 15.2)
        minimal_threshold = 0.3
    elif "Seawater_temperature" == data_source:
        value_range_config = (-2, 32)
        minimal_threshold = 0.02
    else:
        print('Unsupported! You need to find out the hyper-parameters by yourselves.')
        pass
        
    return predictor_config, value_range_config, minimal_threshold
    
