import matplotlib.pyplot as plt
import os
import numpy as np
import time
from itertools import chain

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


class Trace(Callback):

    def __init__(self, n, sample_fn, max_length=2000, epoch_thinout=1, columns=2, log_buffers=True):

        self.n = n

        self.sample_fn = sample_fn

        self.epoch_thinout = epoch_thinout
        self.columns = columns

        self.batches_per_epoch = 1

        self.metrics_names = set()
        self.max_length = max_length

        self.log_buffers = log_buffers

        super(Trace, self).__init__()

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.model = pl_module
        if self.log_buffers:
            self.weight_names = [name for name, _ in chain(self.model.named_parameters(), self.model.named_buffers())]
        else:
            self.weight_names = [name for name, _ in self.model.named_parameters()]

        self.traces = {}
        self.trace_length = 0
        self.name = 'logs'#trainer.logger.log_dir
        self.epoch = 0
        self.tick = 0
        self.thinout = 1

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        pass

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module, outputs):

        #self.add_metrics_to_traces(self.batches_per_epoch, logs)
        if self.epoch % self.epoch_thinout == 0:
            self.plot_all()

        self.epoch += 1

    @rank_zero_only
    def on_train_batch_end( self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        self.batches_per_epoch = max(self.batches_per_epoch, batch_idx)

        def some_weights(w, n):
            sz = w.size
            if sz > n:
                return w.flatten()[np.round(np.linspace(0, sz - 1, n)).astype(np.int)]
            else:
                return w.flatten()

        if self.tick % self.thinout == 0:
            for name, w in chain(self.model.named_parameters(), self.model.named_buffers()):
                # w = w.detach().cpu().numpy()
                w = w.data.cpu().numpy()
                # w = np.random.randn(100,200)
                v = some_weights(w, self.n)
                item = self.traces.setdefault(name, {'values': [], 'batch': [], 'epoch': []})
                item['values'].append(v)
                item['batch'].append(batch_idx)
                item['epoch'].append(self.epoch)
            self.trace_length += 1

            # self.add_metrics_to_traces(batch_idx, logs)
        self.tick += 1

        if self.trace_length % self.max_length == 0 :
            for name, item in self.traces.items():
                item['values'] = item['values'][::2]
                item['batch'] = item['batch'][::2]
                item['epoch'] = item['epoch'][::2]
            self.trace_length //= 2
            self.thinout *= 2

        if batch_idx % 100 == 0:
            self.plot_all()


    # def add_metrics_to_traces(self, batch, logs):
    #     logs = logs or {}
    #     for name, v in logs.items():
    #         item = self.traces.setdefault(name, {'values': [], 'batch': [], 'epoch': []})
    #         item['values'].append(v)
    #         item['batch'].append(batch)
    #         item['epoch'].append(self.epoch)
    #
    #     self.metrics_names = self.metrics_names.union(
    #         {k.replace('val_', '') for k, v in logs.items() if k.startswith('val_')})
    #     self.metrics_names = self.metrics_names.union(
    #         {k for k, v in logs.items() if not k.startswith('val_') and k is not 'size' and k is not 'batch'})

    def plot_all(self, use_MPLD3=False):

        for plot_fn, descr in [(self.plot_all_items, 'weights'),
                               # (self.plot_all_metrics, 'metrics'),
                               (self.plot_all_samples, 'samples')
                               ]:
            try:
                fig = plot_fn(self.batches_per_epoch)

                filename = self.name + '/' + descr + ('.html' if use_MPLD3 else '.png')
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                if use_MPLD3:
                    with open(filename, 'w') as file:
                        mpld3.save_html(fig, file)
                else:
                    plt.savefig(filename, bbox_inches='tight')

                plt.close(fig)
            except TypeError:  # in particular catch missing sample_fn
                pass

    def plot_item(self, ax, item, batches_per_epoch):
        x = np.array(item['batch'], dtype='double') / batches_per_epoch + np.array(item['epoch'])
        y = np.stack(item['values'])

        ylim = np.percentile(y, [2.5, 97.5])
        ylim = [ylim[0] - 0.1 / 0.95 * (ylim[1] - ylim[0]), ylim[1] + 0.1 / 0.95 * (ylim[1] - ylim[0])]
        if ax.lines:
            ylim_ = ax.get_ylim()
            ylim = [min(ylim[0], ylim_[0]), max(ylim[1], ylim_[1])]

        if x.shape[0] < 1000:
            _ = ax.plot(x, y, '-')
            ax.set_ylim(*ylim)
        else:
            i = np.round(np.linspace(0, x.shape[0] - 1, 1000)).astype(np.int32)
            _ = ax.plot(x[i], y[i], '-')
            ax.set_ylim(*ylim)

    def plot_all_items(self, batches_per_epoch):

        M = self.columns
        N = len(self.weight_names) // M + 1

        fig = plt.figure(figsize=(6 * M, 4 * N))
        fig.clf()
        fig.subplots(nrows=N, ncols=M)
        # for i,(name,item) in enumerate(sorted(self.traces.iteritems())):
        for i, name in enumerate(sorted(self.weight_names)):
            try:
                item = self.traces[name]
                ax = plt.subplot(N, M, i + 1)
                self.plot_item(ax, item, batches_per_epoch)
                plt.title(name)
            except:
                pass

        ax = plt.axes([0, 0.9, 1.0, 0.1], frameon=False)
        # ax.axis('off')
        ax.grid(False)
        # ax.set_axis_off()
        # ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, self.name + ' \nweight traces \n' + time.strftime("%c"), ha='center', fontsize=24)

        return fig

    # def plot_all_metrics(self, batches_per_epoch):
    #
    #     M = 2
    #     N = (len(self.metrics_names) + 1) // M
    #
    #     fig = plt.figure(figsize=(6 * M, 4 * N))
    #     fig.clf()
    #     fig.subplots(nrows=N, ncols=M)
    #     # for i,(name,item) in enumerate(sorted(self.traces.iteritems())):
    #     for i, name in enumerate(sorted(self.metrics_names)):
    #         for name_ in [name, 'val_' + name]:
    #             try:
    #                 item = self.traces[name_]
    #                 ax = plt.subplot(N, M, i + 1)
    #                 self.plot_item(ax, item, batches_per_epoch)
    #                 ax.grid(True)
    #                 # ax.set_xscale('log')
    #
    #             except:
    #                 pass
    #         plt.title(name)
    #
    #     ax = plt.axes([0, 0.9, 1.0, 0.1], frameon=False)
    #     # ax.axis('off')
    #     ax.grid(False)
    #     ax.set_xscale('log')
    #     # ax.set_axis_off()
    #     # ax.set_frame_on(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.text(0.5, 0.5, self.name + ' \nmetrics \n' + time.strftime("%c"), ha='center', fontsize=24)
    #
    #     return fig

    def plot_all_samples(self, batches_per_epoch):

        fig = self.sample_fn()

        ax = plt.axes([0, 0.9, 1.0, 0.1], frameon=False)
        # ax.axis('off')
        ax.grid(False)
        # ax.set_axis_off()
        # ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, self.name + ' \nsamples \n' + time.strftime("%c"), ha='center', fontsize=24)

        return fig


