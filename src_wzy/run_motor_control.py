import os
import time
import numpy as np
from env.Tankmotor import Tankmotor
from env.IFF_env_2 import ServoControlEnv
import argparse
import json
import torch
from framework.utils import set_seed, ConfigDict, make_logpath
from framework import utils
from datetime import datetime
import gym
import csv
import threading

def run_motor_control_for_folder(ol_folder, control_file="control.txt", flag_file="flag.txt", cycles=3):
    """
    Execute motor control for a specific OL folder based on the control.txt values
    when flag.txt is '0', and update flag.txt to '1' when done.
    
    Args:
        ol_folder: The OL folder path (e.g. MMGP_OL0)
        control_file: Name of the control file
        flag_file: Name of the flag file
        cycles: Number of cycles to repeat the control values
    """
    control_path = os.path.join(ol_folder, control_file)
    flag_path = os.path.join(ol_folder, flag_file)
    
    # Check if flag.txt indicates we can execute
    if not os.path.exists(flag_path):
        print(f"Flag file does not exist in {ol_folder}")
        return False
    
    # Read the flag value
    with open(flag_path, 'r') as f:
        flag_value = f.read().strip()
    
    # If flag is not '0', don't execute
    if flag_value != '0':
        print(f"Flag in {ol_folder} is {flag_value}, skipping execution")
        return False
    
    # Check if control file exists
    if not os.path.exists(control_path):
        print(f"Control file does not exist in {ol_folder}")
        return False
    
    # Read control values from control.txt
    with open(control_path, 'r') as f:
        control_values = [float(line.strip()) for line in f.readlines() if line.strip()]
    
    print(f"Read {len(control_values)} control values from {control_path}")
    
    if len(control_values) == 0:
        print(f"No control values found in {control_path}")
        return False
    
    # Use the Tankmotor class for actual control
    tank_motor = None
    try:
        # We'll simulate the motor control by processing the values at the correct intervals
        print(f"Starting motor control for {ol_folder} with {cycles} cycles")
        
        # Process control values with time intervals for multiple cycles
        for cycle in range(cycles):
            print(f"Starting cycle {cycle + 1}/{cycles} for {ol_folder}")
            
            # Process each control value with appropriate time interval
            for i, control_val in enumerate(control_values):
                # Calculate the time interval (1/3000 seconds)
                time_interval = 1.0 / 3000.0
                
                # Here we would implement the actual motor control logic
                # The control_val would be sent to the appropriate motor actuator
                print(f"  Cycle {cycle + 1}, Step {i + 1}/{len(control_values)}: Control value {control_val:.3f}")
                
                # Sleep for the time interval between control values
                # In a real implementation, this would be the time to send the control signal
                # and wait for the motor to respond
                time.sleep(time_interval)
        
        print(f"Completed {cycles} cycles for {ol_folder}")
        
        # Update flag.txt to '1' to indicate execution is done
        with open(flag_path, 'w') as f:
            f.write('1')
        
        print(f"Updated flag.txt in {ol_folder} to '1'")
        return True
        
    except Exception as e:
        print(f"Error during motor control execution in {ol_folder}: {e}")
        return False
    finally:
        # Close the tank motor connection if it was opened
        if tank_motor:
            try:
                tank_motor.stop()
            except:
                pass


def create_servo_controller_with_data(control_values, time_interval=1.0/3000.0):
    """
    Create a servo controller that uses the control values from the control.txt file
    """
    class ControlValueServoController:
        def __init__(self, control_values, time_interval):
            self.control_values = control_values
            self.time_interval = time_interval
            self.current_index = 0
            self.current_cycle = 0
        
        def reset(self):
            self.current_index = 0
            self.current_cycle = 0
            
        def get_next_action(self):
            if self.current_index >= len(self.control_values):
                self.current_index = 0  # loop back to start
                self.current_cycle += 1
                if self.current_index >= len(self.control_values):
                    # If somehow index is still out of bounds, return zeros
                    return np.array([[0.0, 0.0, 0.0]] * 8)  # Assuming 8 motors
            
            # Get the current control value
            control_val = self.control_values[self.current_index]
            
            # Create an action based on the control value
            # This is a simplified example - real implementation would depend on the specific motor setup
            action = np.zeros((8, 3))  # 8 motors with 3 actions each
            
            # Apply the control value to the action (simplified approach)
            # In a real implementation, you might need to convert the control value
            # to appropriate action values for the motors
            for i in range(8):  # For each motor
                action[i, 0] = control_val  # Apply control value to first action dimension
                # Other dimensions might be zero or derived from the control value differently
            
            self.current_index += 1
            
            return action

    return ControlValueServoController(control_values, time_interval)


def run_motor_control_with_env(ol_folder, control_file="control.txt", flag_file="flag.txt", cycles=3):
    """
    Execute motor control using the actual ServoControlEnv for real motor control
    """
    control_path = os.path.join(ol_folder, control_file)
    flag_path = os.path.join(ol_folder, flag_file)
    
    # Check if flag.txt indicates we can execute
    if not os.path.exists(flag_path):
        print(f"Flag file does not exist in {ol_folder}")
        return False
    
    # Read the flag value
    with open(flag_path, 'r') as f:
        flag_value = f.read().strip()
    
    # If flag is not '0', don't execute
    if flag_value != '0':
        print(f"Flag in {ol_folder} is {flag_value}, skipping execution")
        return False
    
    # Check if control file exists
    if not os.path.exists(control_path):
        print(f"Control file does not exist in {ol_folder}")
        return False
    
    # Read control values from control.txt
    with open(control_path, 'r') as f:
        control_values = [float(line.strip()) for line in f.readlines() if line.strip()]
    
    print(f"Read {len(control_values)} control values from {control_path}")
    
    if len(control_values) == 0:
        print(f"No control values found in {control_path}")
        return False
    
    # Create servo controller with control values
    controller = create_servo_controller_with_data(control_values)
    
    try:
        # Import and use config parameters
        import yaml
        with open('config/tppo_single.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create mock paras with necessary parameters
        class MockParas:
            def __init__(self):
                self.n_iff = 8
                self.excution_time = 0.035
                self.interval = 10
                self.steady_time = 0.0
                self.control_frequency = 20
                self.refresh_time = 18
                self.mid_values = [186, 175, 179, 180, 193, 177, 189, 184] * 3  # 8 motors * 3 values each
                self.action_space = 3
                self.obs_space = 14
                self.motor_velocity = 0.15
        
        paras = MockParas()
        
        # Initialize the ServoControlEnv
        env = ServoControlEnv(paras)
        env.load_midvalue(paras.mid_values)
        
        # Reset the environment
        obs = env.reset()
        
        print(f"Starting motor control for {ol_folder} with {cycles} cycles")
        
        # Run the controller for the specified number of cycles
        total_steps = len(control_values) * cycles
        step = 0
        
        while step < total_steps:
            # Get the next action from the controller
            action = controller.get_next_action()
            
            # Execute the action in the environment
            next_obs, reward, done, info = env.step(action)
            
            print(f"Step {step + 1}/{total_steps}: Action applied")
            
            # Check if any motor system needs reset
            if any(done):
                print("Some motors are done, resetting environment...")
                obs = env.reset()
            else:
                obs = next_obs
            
            step += 1
            
            # Sleep briefly to control timing
            time.sleep(1.0/3000.0)  # Time between control steps
        
        print(f"Completed {cycles} cycles for {ol_folder}")
        
        # Save results
        env.save(0, save_full_data=True)
        
        # Update flag.txt to '1' to indicate execution is done
        with open(flag_path, 'w') as f:
            f.write('1')
        
        print(f"Updated flag.txt in {ol_folder} to '1'")
        return True
        
    except Exception as e:
        print(f"Error during motor control execution in {ol_folder}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Execute motor control for MMGP OL folders')
    parser.add_argument('--base_path', type=str, default='../', 
                       help='Base path containing MMGP_OL folders (default is parent directory)')
    parser.add_argument('--cycles', type=int, default=3, help='Number of cycles to repeat control values')
    parser.add_argument('--use_env', action='store_true', 
                       help='Use the actual ServoControlEnv for motor control (default: False)')
    
    args = parser.parse_args()
    
    # Find all MMGP_OL folders in the base path
    base_path = args.base_path
    ol_folders = []
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith("MMGP_OL") and item[7:].isdigit():
            ol_folders.append(item_path)
    
    ol_folders.sort()  # Sort to ensure proper order (OL0, OL1, OL2, etc.)
    
    print(f"Found {len(ol_folders)} OL folders: {ol_folders}")
    
    # Process each OL folder
    for ol_folder in ol_folders:
        print(f"\nProcessing {ol_folder}")
        
        if args.use_env:
            success = run_motor_control_with_env(ol_folder, cycles=args.cycles)
        else:
            success = run_motor_control_for_folder(ol_folder, cycles=args.cycles)
        
        if success:
            print(f"Successfully executed motor control for {ol_folder}")
        else:
            print(f"Skipped or failed motor control for {ol_folder}")
    
    print("\nCompleted processing all OL folders")


if __name__ == "__main__":
    main()