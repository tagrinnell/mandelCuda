# 
#   Plotting points from csv using pandas dataframes + seaborn scatterplot for Mandlebrot Set. 
#   
#   Author: Tasman Grinnell
#

import seaborn
import pandas
import matplotlib.pyplot as plt

def main() :
    # Process input
    # Laptop
    # csv = pandas.read_csv("C:\\Users\\Devil\\Desktop\\Random Docs\\CudaProgramming\\mandelCuda\\mandelCudaSequential\\MandelSetOut.txt")
    
    # PC
    csv = pandas.read_csv("C:\\Users\\tasma\\Desktop\\Textbooks\\mandelCuda\\CSVOutputs\\MandelSetOut.csv")

    plot = seaborn.scatterplot(data=csv, x='X', y='Y', hue='Iteration', linewidth=0, legend=False)
    plot.set_xlabel('')
    plot.set_ylabel('')
    plot.set_xticklabels([])
    plot.set_yticklabels([])
    
    plt.show()
    return 0

if __name__ == "__main__" :
    main()  