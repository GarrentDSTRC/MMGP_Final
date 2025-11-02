#!/usr/bin/env python
"""
Main script to run the motor control system for MMGP_OL folders.
This script will monitor flag.txt files and execute control.txt when flag is '0',
then update flag to '1'.
"""

import sys
import os

# Add the parent directory to the path so we can import from the parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_control_manager import MotorControlManager
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run motor control system for MMGP OL folders')
    parser.add_argument('--base_path', type=str, default='../', 
                       help='Base path containing MMGP_OL folders (default is parent directory)')
    parser.add_argument('--cycles', type=int, default=3, help='Number of cycles to repeat control values')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuously monitoring for new control requests')
    parser.add_argument('--check_interval', type=float, default=1.0,
                       help='Check interval in seconds when running continuously')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode without actual motor control')
    
    args = parser.parse_args()
    
    print("Initializing Motor Control System...")
    print(f"Base path: {args.base_path}")
    print(f"Cycles: {args.cycles}")
    print(f"Continuous mode: {args.continuous}")
    
    # Create motor control manager
    manager = MotorControlManager(base_path=args.base_path, cycles=args.cycles)
    
    if args.continuous:
        print(f"Starting continuous monitoring (check interval: {args.check_interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            manager.monitor_and_execute(check_interval=args.check_interval)
        except KeyboardInterrupt:
            print("\nStopping continuous monitoring...")
            manager.stop()
    else:
        print("Running single execution cycle...")
        manager.execute_once()
        print("Motor control execution completed.")


if __name__ == "__main__":
    main()