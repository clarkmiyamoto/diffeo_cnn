import os
import inspect

def pwd() -> 'os':
    """
    Get present working directory of caller script
    
    Returns:
        caller_dir_name
    """
    # Get the caller script's directory
    frame = inspect.stack()[1]
    caller_script_path = frame.filename
    caller_directory = os.path.dirname(os.path.abspath(caller_script_path))
    caller_dir_name = os.path.basename(caller_directory)
    
    return caller_dir_name