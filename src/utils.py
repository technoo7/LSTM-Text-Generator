import configparser

def read(file_name):
    """
    Reads a text that is in the data folder and returns the text
    
    Args:
        file_name (str): a name of the file that is in the data folder that needs to be read.
        
    Returns:
        text (str): the data of the file
    
    """
    with open(f'data//{file_name}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_config_value(config_file_path, section_name, key_name):
    """
    Read a configuration file and return a dictionary of settings.

    Args:
        config_file_path (str): Path to the configuration file.
        section_name (str): Section of the configuration file.
        key_name (str): Key that's value needed.

    Returns:
        value (str): a string containing value.
    """
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the .conf file
    config.read(config_file_path)

    # Access values in the .conf file
    value = config.get(section_name, key_name)

    return value
