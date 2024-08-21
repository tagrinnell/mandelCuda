#
#
#   Plotting from CSV directly through a csv dataframe read
#
#   Author: Tasman Grinnell
#

import matplotlib.pyplot as plt
import pandas

def main() :

    csv = pandas.read_csv("C:\\Users\\tasma\\Desktop\\Textbooks\\mandelCuda\\CSVOutputs\\MandelSetOut_Parallel.csv")

    # fig, ax = plt.subplots()

    # keys = csv.keys()
    # ax.scatter(csv["X", "Y"])

    # colormaps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    ax = csv.plot.scatter(x="X", y="Y", c="Iteration", colormap='magma_r')
    
    frame1 = plt.gca()

    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    plt.show()

    return 0

if __name__ == "__main__" :
    main()