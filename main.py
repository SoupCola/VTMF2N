from train import train_net
import sys
import os
import yaml

# - Run `python main.py` to use the default config file: config.yaml
# - Or run `python main.py your_config.yaml` to specify a custom config file

if __name__ == "__main__":
    config_file = 'config.yaml'  # Default configuration file (assumed to be in the current directory)

    # If a command-line argument is provided, use it as the config file path
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print("Running experiment with config file:", config_file)

    # Load the YAML configuration if the file exists
    if os.path.exists(config_file):
        with open(config_file) as stream:
            config = yaml.safe_load(stream)
    else:
        print("Error: Config file does not exist!")
        sys.exit()

    # Create the main output directory if it doesn't exist
    if not os.path.exists(config['save_dir']):
        os.mkdir(config['save_dir'])

    # Create a subdirectory for this specific model: save_dir/Model_name
    model_save_path = config['save_dir'] + '/' + config['Model_name']
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # Start training with the loaded configuration
    train_net(config)