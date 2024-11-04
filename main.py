import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classifier for Selected Model.")
    parser.add_argument('--model', type=str, default='Pengi', help="Pengi")
    parser.add_argument('--model_cfg', type=str, default='./configs/pengi_linear_classifier.yaml' ,help="path to model cfg")
    parser.add_argument('--classifier', type=str, default= 'linear', help="implemented classifiers")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help="Directory containing entailment data csv")
    args = parser.parse_args()

    # Open and read the YAML file
    cfg = OmegaConf.load(args.model_cfg)

    if args.classifier.lower() not in ['linear'] :
        print(f"{args.classifier} is not implemented")
    else :
        if args.model.lower() == 'pengi' :
            print("Task : Training Classifier for Pengi Model")
            if args.classifier.lower() == 'linear' :
                from trainers.pengi_classifier import PengiClassifierTrainer
                trainer = PengiClassifierTrainer(cfg)
        else :
            print(f"{args.model} classifier trainer is not implemented")
