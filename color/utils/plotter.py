import visdom

class MSEPlotter:

    def __init__(self, name='Unnamed', env=None):
        self.vis = visdom.Visdom()
        self.env = 'main' if env is None else env
        self.vis_win = self.vis.line(Y=[[0, 0]], X=[[1, 1]], env=env, opts={
            'width': 1000,
            'title': 'Error Plots - {}'.format(name),
            'xlabel': 'Epoch',
            'ylabel': 'MSE Loss',
            'legend': [
                'Training Loss',
                'CV Loss'
            ]
        })
        self.first = True

    def plot(self, train_loss, cv_loss, epoch):
        update = 'replace' if self.first else 'append'
        self.vis.line(Y=[[float(train_loss), float(cv_loss)]], X=[[epoch, epoch]],
                 env=self.env, win=self.vis_win, update=update)
        self.first = False
