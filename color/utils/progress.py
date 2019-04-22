import sys
import time


class ProgressBar:
    """Terminal progress bar
    Arguments:
        total: Relative size corresponding to completed task
        status: In progress status message
        bar_len: Actual size of progress bar when full
        max_len: Anything beyond this is truncated
        update_freq: Write to terminal at this rate (increase to save time on i/o)
    """

    def __init__(self, total, status='', **kwargs):
        self.total = total
        self.status = status
        self.params = {
            'max_len': 150,
            'bar_len': 60,
            'update_freq': 2
        }
        self.params.update(kwargs)
        self.update_time = None
        self.completed = False

    def update(self, count):
        """Updates the progress bar with a new progress value"""

        curr_time = time.time()
        if self.update_time is not None and self.update_time + self.params['update_freq'] > curr_time:
            return
        self.update_time = curr_time

        filled_len = self.params['bar_len'] if self.total == 0 else \
            int(round(self.params['bar_len'] * count / float(self.total)))
        percents = 100.0 if self.total == 0 else round(100.0 * count / float(self.total), 1)

        clear = ' ' * self.params['max_len']  # Blank line is used to clear previous content on the terminal
        bar = '=' * filled_len + '-' * (self.params['bar_len'] - filled_len)
        output = ('[%s] %s%s ...%s' % (bar, percents, '%', self.status))
        if len(output) > self.params['max_len']:
            output = output[:self.params['max_len']]

        sys.stdout.write('%s\r' % clear)
        sys.stdout.write(output)
        sys.stdout.write('\n' if self.completed else '\r')
        sys.stdout.flush()

    def complete(self, status=''):
        """Fills the progress bar to completion and updates the status message"""
        self.completed = True
        self.status = status
        self.update_time = None
        self.update(self.total)
