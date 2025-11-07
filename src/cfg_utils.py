import datetime
import yaml
import argparse

def load_yaml_config(yaml_file):
    cfg = yaml.safe_load(open(yaml_file))
    # Create filename if we want to save the statistics
    if cfg["save_stats"]["enabled"]:
        filename = cfg["save_stats"]["filename"] + ".pt"
    else: 
        filename = None
    # loss weights
    weights = cfg["weights"]
    obj_weight = float(weights["w_obj"])
    slack_weight = float(weights["w_slack"])
    constraint_weight = float(weights["w_con"])
    supervised_weight = float(weights["w_sup"])
    print(f"Using loss weights - Obj: {obj_weight}, Slack: {slack_weight}, Constraint: {constraint_weight}, Supervised: {supervised_weight}")
    loss_weights = [obj_weight, slack_weight, constraint_weight, supervised_weight]
    # Training params 
    training_params = cfg["training"]
    training_params['TRAINING_EPOCHS'] = int(training_params['TRAINING_EPOCHS'])
    training_params['CHECKPOINT_AFTER'] = int(training_params['CHECKPOINT_AFTER'])
    training_params['LEARNING_RATE'] = float(training_params['LEARNING_RATE'])
    training_params['WEIGHT_DECAY'] =  float(training_params['WEIGHT_DECAY'])
    training_params['PATIENCE'] = int(training_params['PATIENCE'])
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    training_params['RUN_NAME'] = "ssl_" + dt_string
    return filename, loss_weights, training_params

def load_argparse_config():
    parser = argparse.ArgumentParser()
    # Weights
    parser.add_argument("--w_obj", type=float, default=0.0, help="Weight for objective value")
    parser.add_argument("--w_slack", type=float, default=0.0, help="Weight for slack penalty")
    parser.add_argument("--w_con", type=float, default=0.0, help="Weight for constraint violation penalty")
    parser.add_argument("--w_sup", type=float, default=1.0, help="Weight for supervised loss")
    # Training parameters
    parser.add_argument("--TRAINING_EPOCHS", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--CHECKPOINT_AFTER", type=int, default=50, help="Number of steps between checkpoints")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--PATIENCE", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--WANDB_PROJECT", type=str, default="l2o_ssl_miqp_robot_nav", help="WandB project name")
    parser.add_argument("--RUN_NAME_PREFIX", type=str, default="ssl_", help="Prefix for run name")
    # Save statistics
    parser.add_argument("--save_stats", action="store_true", help="Enable saving training statistics")
    parser.add_argument("--filename", type=str, default="robot_nav", help="Filename for saved stats")

    arg = parser.parse_args()
    loss_weights = [arg.w_obj, arg.w_slack, arg.w_con, arg.w_sup]
    training_params = {}
    training_params['TRAINING_EPOCHS'] = int(arg.TRAINING_EPOCHS)
    training_params['CHECKPOINT_AFTER'] = int(arg.CHECKPOINT_AFTER)
    training_params['LEARNING_RATE'] = float(arg.LEARNING_RATE)
    training_params['WEIGHT_DECAY'] =  float(arg.WEIGHT_DECAY)
    training_params['PATIENCE'] = int(arg.PATIENCE)
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    training_params['RUN_NAME'] = "ssl_" + dt_string
    training_params['WANDB_PROJECT'] = arg.WANDB_PROJECT
    training_params['RUN_NAME_PREFIX'] = arg.RUN_NAME_PREFIX

    filename = arg.filename if arg.save_stats else None

    return filename, loss_weights, training_params