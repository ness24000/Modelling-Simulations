import numpy as np

from pyics import Model


def decimal_to_base_k(decimal: int, k: int) -> list[int]:
    number = []
    while decimal > 0:
        remainder = (decimal % k)
        decimal = (decimal // k)
        number.append(remainder)
    number.reverse()
    return number


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))


    def build_rule_set(self):
        len_k = (self.k ** (2*self.r + 1))
        number = decimal_to_base_k(self.rule, self.k)
        while len(number) < len_k:
            number.insert(0, 0)

        self.rule_set = number

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        #transfor input to decimal
        inp = np.flip(inp)
        decimal = 0
        for n in range(len(inp)):
            decimal += inp[n]*(self.k**n)

        #index rule_list with decimal
        output = self.rule_set[int(len(self.rule_set) - (decimal+1))]

        return output

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        init_row = np.zeros(self.width)
        if self.width % 2 != 0:
            init_row[int(self.width/2 + 1/2)-1] = 1
        else:
            init_row[int(self.width/2)-1] = 1

        return init_row

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""

        #count time and stop if height reached
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.

            #obtain the indices of the patch and the relevant neighbors
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]

            #get values of patch and neighbors from row t-1
            values = self.config[self.t - 1, indices]

            #modify row t and column patch based on check_rule
            self.config[self.t, patch] = self.check_rule(values)


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
