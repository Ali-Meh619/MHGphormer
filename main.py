import os
import torch
from src.config import ARGS, DEVICE
from src.dataset import generate_datasets_and_args
from src.models import SeHGNN
from src.trainer import Trainer, save_model

def main():
    print("Generating dataset and computing features...")
    args, train_loader, valid_loader, test_loader = generate_datasets_and_args(ARGS)
    
    print("Initializing SeHGNN model...")
    model = SeHGNN(
        args,
        node_1=args["num_users"],
        feature_1=args["feature_user"],
        node_2=1,
        feature_2=args["feature_BS"],
        node_3=1,
        feature_3=args["feature_IRS"],
        hidden=args["hidden"],
        dropout=args["dropout"]
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=args['lr_init'], 
        eps=1.0e-8,
        weight_decay=1e-5, 
        amsgrad=False
    )
    
    lr_scheduler = None
    if args['lr_decay']:
        print('Applying learning rate decay.')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[
                int(0.33 * args['epochs']),
                int(0.5 * args['epochs']),
                int(0.8 * args['epochs']),
                int(0.9 * args['epochs'])
            ],
            gamma=0.1
        )

    print("Starting training...")
    trainer = Trainer(model, optimizer, train_loader, valid_loader, test_loader, args, lr_scheduler)
    trainer.train()

    print("Saving model...")
    result_train_file = os.path.join("RIS-MIMO-THz")
    save_model(trainer.model, result_train_file, 1)

    print("Running test...")
    total_sum, rate, power, phase, band = trainer.test()
    print("Testing complete. Results shape - Rate:", rate.shape, "Power:", power.shape, "Phase:", phase.shape)

if __name__ == "__main__":
    main()
