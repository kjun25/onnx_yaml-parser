import argparse
import os

import yaml


# Command line options.
parser = argparse.ArgumentParser(
    usage="Helper script to print the histogram from a Glow YAML profile."
)
parser.add_argument('--debug', action='store_true', help='debugging mode')
parser.add_argument(
    "-f", "--file", dest="file", required=True, type=str, help="Profile YAML file path."
)
parser.add_argument(
    "-n",
    "--name",
    dest="name",
    required=True,
    type=str,
    help="Node value name to plot.",
)
parser.add_argument(
    "-l",
    "--log-scale",
    dest="log_scale",
    required=False,
    default=False,
    action="store_true",
    help="Plot the histogram on a logarithmic scale (base 10).",
)

args = parser.parse_args()

# Get arguments.
profile = args.file
name = args.name
log_scale = args.log_scale

# Verify profile exists.
if not os.path.isfile(profile):
    print('File "%s" not found!' % profile)
    exit(1)

# Read YAML data.
print('Reading file "%s" ...' % profile)
data = None
with open(profile, "r") as stream:
    try:
        data = yaml.safe_load_all(stream)
    except yaml.YAMLError as err:
        print(err)

# Search YAML entry for node value.
    print('Searching node value name "%s" ...' % name)
    entry = None
    for item in data:
        if item["NodeOutputName"] == name:
            entry = item

    if not entry:
        print('Node value "%s" not found!' % name)
        exit(1)
