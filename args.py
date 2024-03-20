import argparse
import os

from global_settings import dataset_path, log_path
from utils import load_pkl


def init_arguments():
    """
    define general hyperparams
    """
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--gpu", type=str, default="0", help="index of available GPUs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_sch",
        choices=["cosion", "plateau", "none"],
        default="plateau",
        help="Learning rate scheduler",
    )
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--fc_dropout", type=float, default=0.6, help="dropout rate for final MLP")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--early_stop", type=int, default=4)
    parser.add_argument("--save_model", type=bool, default=False)

    # classification loss
    parser.add_argument(
        "--loss_name",
        choices=["focal_loss", "ce"],
        default="focal_loss",
        help="choosing loss function.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.1,
        help="the weight for the class 0 in the focal loss. default: [0.1, 0.9].",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2,
        help="gamma in the focal loss.",
    )
    parser.add_argument(
        "--bce_loss_weight",
        type=int,
        default=8,
        help="the weight for the class 1 in the bce loss. default: [1, 8].",
    )

    # Dimensions of embeddings
    parser.add_argument(
        "--code_embed_dim",
        type=int,
        default=256,
        help="Dimension of medical codes embeddings.",
    )
    parser.add_argument(
        "--patient_embed_dim",
        type=int,
        default=256,
        help="Dimension of patient embeddings.",
    )

    # dataset
    parser.add_argument("--ignore_infrequent_feat", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fold", type=int, default=0, help="use which fold of train/val/test split.")
    parser.add_argument(
        "--dataset",
        choices=["eicu", "mimic"],
        default="eicu",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_path",
        default=dataset_path,
        type=str,
        help="Dataset file path",
    )

    # log
    parser.add_argument(
        "--wandb_dir",
        default=os.path.join(log_path, "wandb_dir"),
        type=str,
        help="Wandb file path",
    )
    parser.add_argument(
        "--log_dir",
        default=os.path.join(log_path, "log_dir"),
        type=str,
        help="log file and ckpt path",
    )

    # task
    parser.add_argument(
        "--task",
        type=str,
        choices=["ihm", "readmission"],
        default="ihm",
    )
    parser.add_argument(
        "--cluster_evaluate",
        type=bool,
        default=False,
        help="run medical codes clustering experiment.",
    )

    # parser.add_argument(
    #     '--label_ratio',
    #     type=float,
    #     default=1,
    #     help='Ratio of labeled training data in semi-supervised learning setting.',
    # )
    return parser


def add_ssl_config(parser):
    parser.add_argument(
        "--ssl_method",
        type=str,
        choices=["noise_contrast", "barlow", "simclr", "supcon", "none"],
        default="supcon",
    )
    parser.add_argument(
        "--ssl_eva_task",
        type=str,
        choices=["ihm", "readmission"],
        default="ihm",
    )
    parser.add_argument(
        "--ssl_model_name",
        type=str,
        choices=["MVGCL_ssl", "TransGNNssl"],
        default="MVGCL_ssl",
    )
    parser.add_argument(
        "--is_projector",
        default=True,
        type=bool,
        help="whether to use projector head in ssl experiment.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="adjust ssl loss in supervised learning experiment.",
    )
    parser.add_argument(
        "--projector",
        default="256-128-128",
        type=str,
        help="the size of the projector head.",
    )
    parser.add_argument(
        "--barlow_lambd",
        type=float,
        default=0.0001,
        help="hyperparameter lambd in BarlowTwins.",
    )
    parser.add_argument(
        "--simclr_temper",
        type=float,
        default=0.5,
        help="hyperparameter temperature in SimCLR.",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["row_shuffle", "row_column_shuffle"],
        default="row_column_shuffle",
        help="noise type in Noise Contrast.",
    )

    parser.add_argument(
        "--ssl_k_run",
        type=int,
        default=1,
        help="run linear classifier in k fold of train/test splits.",
    )
    parser.add_argument(
        "--ssl_store_path",
        type=str,
        default="/sda/encounter_data/ssl_store_dir/",
        help="directory for saving outputs of ssl models.",
    )

    return parser


def load_config(config):
    config = vars(config)
    if config["dataset"] == "eicu":
        if config["ignore_infrequent_feat"]:
            config["input_dim"] = 2384
            config["dx_map_path"] = os.path.join(config["dataset_path"], config["dataset"], "small_dx_map.pkl")
            config["proc_map_path"] = os.path.join(config["dataset_path"], config["dataset"], "small_proc_map.pkl")
    elif config["dataset"] == "mimic":
        data_path = os.path.join(
            config["dataset_path"],
            config["dataset"],
            "fold_{}".format(config["fold"]),
            "small_validation_csr.pkl",
        )
        x, _, _ = load_pkl(data_path)
        config["input_dim"] = x.shape[1]
    return config


def load_gnn_config(config):
    if config["gnn_name"] == "gat":
        config["gat_input_dim"] = 128
        config["gat_nhid"] = 64
        config["gat_n_heads"] = 4
        config["gat_n_out_heads"] = 4
        config["gat_attndrop"] = 0.6

    # parser.add_argument('--read_best', action='store_true')
    # parser.add_argument('--tag', type=str)
    # parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--gpus', type=int, default=-1, help='number of available GPUs')

    # parser.add_argument('--clip_grad', type=float, default=0, help='clipping gradient')
    # parser.add_argument('--use_amp', action='store_true')
    # parser.add_argument('--auto_lr', action='store_true')
    # parser.add_argument('--auto_bsz', action='store_true')

    # gat
    # parser.add_argument('--gat_activation', type=str, default='elu')
    # parser.add_argument('--gat_negslope', type=float, default=0.2)
    # parser.add_argument('--gat_residual', action='store_true')
    return config


def add_mlp_config(parser):
    parser.add_argument("--hidden_dim", type=int, default=128)
    return parser


def add_transformer_config(parser):
    parser.add_argument("--tran_attn_dropout", type=float, default=0.4)
    parser.add_argument("--tran_num_head", type=int, default=4)
    parser.add_argument("--tran_num_layer", type=int, default=2)
    parser.add_argument(
        "--tran_agg_type",
        type=str,
        choices=["mean", "attn"],
        default="mean",
    )
    return parser


def add_fc_config(parser):
    parser.add_argument("--fc_num_layer", type=int, default=2)
    return parser


def add_trans_gnn_config(parser):
    parser = add_transformer_config(parser)
    parser = add_gnn_config(parser)
    parser.add_argument(
        "--patient_agg_type",
        type=str,
        choices=["attn", "concat", "drop"],
        default="attn",
    )
    return parser


def add_hypergraph_config(parser):
    parser.add_argument(
        "--hgnn_fusion_dim_div",
        type=int,
        default=4,
        help="hidden_dim of fusion in HypergraphLayer is set to embed_dim // hgnn_fusion_dim_div",
    )
    parser.add_argument("--hgnn_num_layer", type=int, default=3)
    parser.add_argument(
        "--hgnn_agg_type",
        type=str,
        choices=["mean", "attn"],
        default="attn",
    )
    parser.add_argument(
        "--hgnn_activation",
        type=str,
        choices=["relu", "leakyrelu", "elu", "tanh", "sigmoid", "none"],
        default="tanh",
    )
    return parser


def add_hyper_trans_config(parser):
    parser = add_transformer_config(parser)
    parser = add_hypergraph_config(parser)
    parser = add_ssl_config(parser)
    parser.add_argument(
        "--patient_agg_type",
        type=str,
        choices=["attn", "concat", "drop"],
        default="concat",
    )
    return parser


def add_hyper_trans_gnn_config(parser):
    parser = add_transformer_config(parser)
    parser = add_hypergraph_config(parser)
    parser = add_gnn_config(parser)
    parser = add_ssl_config(parser)
    parser.add_argument(
        "--patient_agg_type",
        type=str,
        choices=["attn", "concat", "drop"],
        default="attn",
    )
    return parser


def add_gnn_config(parser):
    # patient graph construction
    parser.add_argument(
        "--k_neighbour",
        type=int,
        default=5,
        choices=[3, 5, 7, 9, 11, 13, 15],
        help="Chose number of neighbours when contructing patient graph.",
    )

    # subgraph sampling
    parser.add_argument(
        "--sample_size1",
        type=int,
        default=5,
        help="Number of 1st order neighbours in neighbour sampler.",
    )
    parser.add_argument(
        "--sample_size2",
        type=int,
        default=5,
        help="Number of 2nd order neighbours in neighbour sampler.",
    )
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_featdrop", type=float, default=0.6)

    # choosing model
    parser.add_argument(
        "--gnn_name",
        type=str,
        default="gat",
        choices=["gat", "sage"],
        help="Chose model for graph-level patient embedding.",
    )
    parser.add_argument(
        "--gnn_merge_type",
        type=str,
        default="attn",
        choices=["mean", "attn"],
        help="How to get patient embeddings from codes embeddings.",
    )
    return parser
