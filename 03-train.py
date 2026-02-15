# TODO: 저장을 ckpt 폴더에 맞게 해서 inference 폴더 또는 reproduce 폴더와 일치하게.
"""
data directory, hyperparams -> training cfg file
training cfg file -> training/validation -> ckpt saving
evaluation by inference code? options 조절이 안되는데 out channels 만 조절하면 되려나
"""
import argparse



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["crystal", "redocked", "p2rank"], required=True, help="Coreset types to evaluate InSiteDTA")
    parser.add_argument("--batch_size", type=int, default=64, help="Bacth size for inference")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    # cfg 있으면 그걸로 입력받아서 학습
    return parser.parse_args()


def create_data_cfg():
    pass

def voxelize

def train_model(model, tr_dataloader, vl_dataloader, train_cfg, data_cfg):
    pass


def main():
    train_cfg = get_arguments()
    
    create_cfg()
    train_model(model, tr_dataloader, vl_dataloader, train_cfg, data_cfg)
    return None


if __name__ == "__main__":
    main()