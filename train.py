import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import albumentations
import cv2

######  Local packages  ######
import learning
import models
import dataloader
import util


def main(args):
    if not args.to_augment:
        my_transforms = albumentations.Compose(
            [
                albumentations.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                albumentations.HorizontalFlip(p=0.3),
                albumentations.RandomBrightnessContrast(p=0.3),
                albumentations.augmentations.geometric.transforms.Affine(
                    scale=(0.97, 1.02), translate_px=(0, 40), shear=(-20, 20), p=0.3
                ),
            ]
        )
    else:
        my_transforms = None

    train_data, val_data, test_data = dataloader.create_dataset(
        data_paths=args.data_paths,
        regex_image_paths=args.regex_image_paths,
        transforms=my_transforms,
    )

    train_loader, val_loader, test_loader = dataloader.dataloader(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        batch_size=args.batch_size,
    )

    model = models.residual_UNET(in_channels=3, out_channels=1, init_features=32)

    criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(), betas=args.betas, lr=args.lr, weight_decay=args.weight_decay
    )

    # Schedular
    lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Loading Model
    if args.ckpt_load_path is not None:
        print("******  Loading Model   ******")
        model, optimizer = util.load_model(
            ckpt_path=args.ckpt_load_path, model=model, optimizer=optimizer
        )

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    # Train the model(Train and Validation Steps)
    model, optimizer = learning.Train_mode(
        model=model,
        device=args.device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        saving_checkpoint_path=args.ckpt_save_path,
        saving_prefix=args.ckpt_prefix,
        saving_checkpoint_freq=args.ckpt_save_freq,
        report_path=args.report_path,
    )

    return model


if __name__ == "__main__":
    args = util.get_args()
    main(args)
