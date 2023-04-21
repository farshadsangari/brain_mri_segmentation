import torch
import pandas as pd
from tqdm import tqdm
import os
import models as models
import util as util


def Train_mode(
    model,
    device,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    saving_checkpoint_path,
    saving_prefix,
    saving_checkpoint_freq,
    report_path,
):

    report = pd.DataFrame(
        columns=[
            "mode",
            "epoch",
            "batch_index",
            "learning_rate",
            "loss_batch",
            "avg_epoch_loss_till_current_batch",
            "avg_epoch_jaccard_till_current_batch",
            "avg_epoch_dice_till_current_batch",
        ]
    )

    ###################################    Training mode     ##########################################
    for epoch in range(1, num_epochs + 1):
        train_loss = util.AverageMeter()
        val_loss = util.AverageMeter()

        train_jaccard = util.AverageMeter()
        val_jaccard = util.AverageMeter()

        train_dice = util.AverageMeter()
        val_dice = util.AverageMeter()

        dice = util.Dice_Coefficient()
        jaccard = util.Jaccard_Metric()

        mode = "train"
        model.train()
        # Loop for train batches
        loop_train = tqdm(
            enumerate(train_dataloader, 1),
            total=len(train_dataloader),
            desc="train",
            position=0,
            leave=True,
        )
        for batch_index, (_, _, inputs, labels) in loop_train:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            jaccard_batch = jaccard(labels, outputs).item()
            dice_batch = dice(outputs, labels).item()

            train_loss.update(loss.item(), inputs.size(0))
            train_jaccard.update(jaccard_batch, inputs.size(0))
            train_dice.update(dice_batch, inputs.size(0))

            new_row = pd.DataFrame(
                {
                    "mode": mode,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss_batch": loss.item(),
                    "avg_epoch_loss_till_current_batch": train_loss.avg,
                    "avg_epoch_jaccard_till_current_batch": train_jaccard.avg,
                    "avg_epoch_dice_till_current_batch": train_dice.avg,
                },
                index=[0],
            )
            report.loc[len(report)] = new_row.values[0]

            loop_train.set_description(f"Train mode - epoch : {epoch}")
            loop_train.set_postfix(
                Loss_Train="{:.4f}".format(train_loss.avg),
                Jaccard_Train="{:.4f}".format(train_jaccard.avg),
                Dice_Train="{:.4f}".format(train_dice.avg),
                refresh=True,
            )
            if (epoch % saving_checkpoint_freq) == 0:
                util.save_model(
                    file_path=saving_checkpoint_path,
                    file_name=f"{saving_prefix}{epoch}.ckpt",
                    model=model,
                    optimizer=optimizer,
                )

        ################################    Validation mode   ##############################################
        model.eval()
        mode = "validation"
        with torch.no_grad():

            # Loop for val batches
            loop_val = tqdm(
                enumerate(val_dataloader, 1),
                total=len(val_dataloader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_index, (_, _, inputs, labels) in loop_val:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                jaccard_batch = jaccard(labels, outputs).item()
                dice_batch = dice(outputs, labels).item()

                val_loss.update(loss.item(), inputs.size(0))
                val_jaccard.update(jaccard_batch, inputs.size(0))
                val_dice.update(dice_batch, inputs.size(0))

                new_row = pd.DataFrame(
                    {
                        "mode": mode,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "loss_batch": loss.item(),
                        "avg_epoch_loss_till_current_batch": val_loss.avg,
                        "avg_epoch_jaccard_till_current_batch": val_jaccard.avg,
                        "avg_epoch_dice_till_current_batch": val_dice.avg,
                    },
                    index=[0],
                )
                report.loc[len(report)] = new_row.values[0]

                optimizer.zero_grad()
                loop_val.set_description(f"Validation mode - epoch : {epoch}")
                loop_val.set_postfix(
                    Loss_val="{:.4f}".format(val_loss.avg),
                    Jaccard_val="{:.4f}".format(val_jaccard.avg),
                    Dice_val="{:.4f}".format(val_dice.avg),
                    refresh=True,
                )
        lr_scheduler.step()

    report.to_csv(os.path.join(report_path, f"report_training.csv"))
    return model, optimizer


def Inference_mode(
    model,
    device,
    test_dataloader,
    criterion,
    optimizer,
):
    test_loss = util.AverageMeter()
    test_jaccard = util.AverageMeter()
    test_dice = util.AverageMeter()
    dice = util.Dice_Coefficient()
    jaccard = util.Jaccard_Metric()

    model.eval()
    with torch.no_grad():
        test_loss = util.AverageMeter()
        # Loop for test batches
        loop_test = tqdm(
            enumerate(test_dataloader, 1), total=len(test_dataloader), desc="val"
        )
        for batch_index, (inputs, labels) in loop_test:

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            jaccard_batch = jaccard(labels, outputs).item()
            dice_batch = dice(outputs, labels).item()

            test_loss.update(loss.item(), inputs.size(0))
            test_jaccard.update(jaccard_batch, inputs.size(0))
            test_dice.update(dice_batch, inputs.size(0))

            loop_test.set_description(f"Test mode")
            loop_test.set_postfix(
                Loss_test="{:.4f}".format(test_loss.avg),
                Jaccard_test="{:.4f}".format(test_jaccard.avg),
                Dice_test="{:.4f}".format(test_dice.avg),
                refresh=True,
            )
        print(f"@the end of training, Test loss value is : {test_loss.avg}")
