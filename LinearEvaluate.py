import os
from pprint import pprint

from args import add_ssl_config, init_arguments, load_config
from models.base import evaluation

parser = init_arguments()
parser = add_ssl_config(parser)
config = parser.parse_args()
config = load_config(config)


# model_name = 'TransGNNssl_{}'.format(config['ssl_method'])
model_name = '{}_{}'.format(config['ssl_model_name'], config['ssl_method'])
config['model_name'] = model_name

config['ssl_load_path'] = os.path.join(
    config['ssl_store_path'],
    '{}_{}_{}'.format(config['dataset'], config['fold'], model_name),
)
# config['ssl_load_path'] = os.path.join(
#     config['ssl_store_path'],
#     '{}_{}'.format(config['dataset'], model_name),
# )

evaluation(config, config['ssl_eva_task'])
