import multiprocessing
import subprocess
import time
import random
from datetime import datetime
import traceback
import os
import argparse

def update_data_src(commands, data_name):
    for i in range(len(commands)):
        commands[i] = commands[i].replace("<<data_name>>", data_name)
        
    return commands
    
def is_done(command):
    formatted_command = command.replace("<<data_name>>", data_name)
    
    start_str_output = formatted_command.find("/results/")
    end_str_output = formatted_command.find(".ts")
    
    if start_str_output >= 0 and end_str_output >= 0:
        output_file = formatted_command[start_str_output:end_str_output] + ".ts"
        output_file = output_file.replace("/results", "./2-results")
        
        if os.path.isfile(output_file):
            return True
            
    start_str_output = formatted_command.find("/results/")
    end_str_output = formatted_command.find(".pkl")
    
    if start_str_output >= 0 and end_str_output >= 0:
        output_file = formatted_command[start_str_output:end_str_output] + ".ts"
        output_file = output_file.replace("/results", "./2-results")
        
        if os.path.isfile(output_file):
            return True
    
    return False
        
    
def logging(message):
    f = open(logfile, "a")
    f.write(f"{datetime.now()}," + message)
    f.close()

def execute(command):
    logging(f"Executing: {command}")
    if is_done(command):
        logging(f"Completed: {command}")
        return
        
    process = subprocess.Popen(command, shell=True)
    try:
        process.wait(timeout=timeout)  # Set the timeout of experiments
        logging(f"Completed: {command}")
    except subprocess.TimeoutExpired:
        process.terminate()  # Terminate the process if timeout is reached
        logging(f"Timeout: {command}")
    except:
        logging(f"Error: {command}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute experiments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--timeout', type=int, default=172800, help='Maximum time each experiment allowed to run (in seconds / default: 48 hours)')
    parser.add_argument('--num_processes', type=int, default=32, help='Number of experiments to be executed in parallel (default: 32)')
    parser.add_argument('alg_type', choices=['unsupervised', 'supervised'], help='Learning types of algorithms to be executed')
    parser.add_argument('data_src', 
                        choices=['Tide_Pressure_benchmark', 'Tide_Pressure_validation', 'Seawater_temperature', 'Wave_height'], 
                        help='Data source to be experimented')
    
    args = parser.parse_args()
    config = vars(args)

    
    num_cores = config['num_processes']
    timeout = config['timeout']
    data_experimented = config['data_src']
    
    with open("4-param_optimization/" + data_experimented + ".txt", "r") as f:
        commands = f.readlines()
    
    data_name = config["data_src"]
    logfile = data_name + ".log"
    
    commands=update_data_src(commands, data_name)

    # Start the programs in parallel using a process pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(execute, commands)

    print("All commands completed.")
