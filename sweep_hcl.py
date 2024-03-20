from args import (add_hyper_trans_gnn_config, init_arguments, load_config,
                  load_gnn_config)
from global_settings import project_name_prefix
from models.base import run_exp
from models.combined_models import HCL

parser = init_arguments()
parser = add_hyper_trans_gnn_config(parser)
config = parser.parse_args()
config = load_config(config)
config = load_gnn_config(config)

if config["ssl_method"] == "none":
    model_name = "HCL/CL-{}-{}k".format(config["gnn_name"], config["k_neighbour"])
else:
    model_name = "HCL-{}-{}k".format(config["gnn_name"], config["k_neighbour"])
model = HCL(config)

run_exp(config, model, model_name, project_name_prefix)
