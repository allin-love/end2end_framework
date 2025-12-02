import os
import shutil

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.base import LoggerCollection


class LogManager(Callback):

    def __init__(self, fallback_log_dir=None):
        super().__init__()
        self.fallback_log_dir = fallback_log_dir

    def _resolve_log_dir(self, logger):
        if logger is None:
            return None

        if isinstance(logger, LoggerCollection):
            for child in logger.loggers:
                path = self._resolve_log_dir(child)
                if path:
                    return path
            return None

        log_dir = getattr(logger, 'log_dir', None)
        if log_dir:
            return log_dir

        save_dir = getattr(logger, 'save_dir', None)
        name = getattr(logger, 'name', None)
        version = getattr(logger, 'version', None)
        if save_dir and name is not None and version is not None:
            return os.path.join(save_dir, name, f'version_{version}')

        return None

    def on_init_end(self, trainer):
        log_dir = self._resolve_log_dir(trainer.logger) or self.fallback_log_dir
        if log_dir is None:
            raise ValueError('Unable to determine log directory for LogManager. Provide fallback_log_dir.')

        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        print("*"*30)
        print('checkpoint_dir', ckpt_dir)
        print("*"*30)
        if os.path.exists(ckpt_dir):
            ckpt_list = [filename for filename in os.listdir(ckpt_dir) if '.ckpt' in filename]
            print(f'Log and the following checkpoint exists:\n Log dir: {log_dir}\n' + '\n'.join(
                f'[{i}] {filename}' for i, filename in enumerate(ckpt_list)))

            delete = ['delete', 'd']
            resume = ['resume', 'r']
            quit = ['quit', 'q']
            ans = ''
            n_files = len(ckpt_list)
            while not (ans in delete or ans in resume or ans in quit):
                ans = input(f'[Number of ckpt files: {n_files}]\n'	
                            f'Delete the existing log and start a new experiment? Resume? Quit? (d/r/q) << ').lower()
                if ans in delete:
                    shutil.rmtree(log_dir)
                    os.makedirs(log_dir)
                elif ans in resume:
                    if n_files == 0:
                        print('Any checkpoint files do not exist!')
                        ans = ''
                    else:
                        s = ''
                        if n_files > 1:
                            while not (s.isdigit() and int(s) in range(n_files)):
                                s = input(f'Select which checkpoint to load. [0-{n_files - 1}]<< ').lower()
                            ckpt_path = os.path.join(ckpt_dir, ckpt_list[int(s)])
                        else:
                            ckpt_path = os.path.join(ckpt_dir, ckpt_list[0])
                        trainer.resume_from_checkpoint = ckpt_path
                elif ans in quit:
                    raise ValueError('Stopped as the log exist for this experiment.')
        else:
            print(f'Starting a new experiment and logging at \n {os.path.expanduser(log_dir)}')

    def on_keyboard_interrupt(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            yes = ['yes', 'y']
            no = ['no', 'n']
            ans = ''
            while not (ans in no or ans in yes):
                ans = input('Save this run? (Y/N) << ').lower()
                log_dir = self._resolve_log_dir(trainer.logger) or self.fallback_log_dir
                if ans in no:
                    shutil.rmtree(log_dir)
                    print(f'Deleted {log_dir}')
                    # Delete ModelCheckpoint callbacks
                    trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)]
                elif ans in yes:
                    ckpt_path = os.path.join(log_dir, 'checkpoints', 'interrupted_model.ckpt')
                    trainer.save_checkpoint(ckpt_path)
                    print('Saved a checkpoint...')
