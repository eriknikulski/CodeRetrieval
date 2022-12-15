import torch

import const
import ddp

BEST_LOSS = float('inf')
LAST_EPOCH = 0
CURRENT_SAVE_ID = 0


def models(encoder, decoder):
    if ddp.is_main_process():
        model(encoder.state_dict(), const.MODEL_ENCODER_PATH)
        model(decoder.state_dict(), const.MODEL_DECODER_PATH)

    print('saved models')


def model(*args, **kwargs):
    if ddp.is_main_process():
        torch.save(*args, **kwargs)


def checkpoint_models(epoch, encoder, encoder_optimizer, decoder, decoder_optimizer, loss, base_path):
    if ddp.is_main_process():
        checkpoint_model(epoch, encoder, encoder_optimizer, loss, base_path + f'encoder_{epoch}.pt')
        checkpoint_model(epoch, decoder, decoder_optimizer, loss, base_path + f'decoder_{epoch}.pt')


def checkpoint_model(epoch, net, optimizer, loss, path):
    if ddp.is_main_process():
        global BEST_LOSS, LAST_EPOCH
        epoch_dist = epoch - LAST_EPOCH
        if loss < BEST_LOSS and epoch >= const.MIN_CHECKPOINT_EPOCH and epoch_dist >= const.MIN_CHECKPOINT_EPOCH_DIST:
            BEST_LOSS = loss
            LAST_EPOCH = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path)
            print(f'saved checkpoint at epoch: {epoch} with loss: {loss}')
