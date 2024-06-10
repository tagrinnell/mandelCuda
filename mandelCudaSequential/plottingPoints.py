#import pip

#pip.main['install', 'seaborn']

import seaborn
import pandas
import matplotlib.pyplot as plt

def main() :

    # Process input
    # Laptop
    # csv = pandas.read_csv("C:\\Users\\Devil\\Desktop\\Random Docs\\CudaProgramming\\mandelCuda\\mandelCudaSequential\\MandelSetOut.txt")
    csv = pandas.read_csv("C:\\Users\\tasma\\Desktop\\Textbooks\\mandelCuda\\mandelCudaSequential\\MandelSetOut.csv")
    plot = seaborn.scatterplot(data=csv, x='X', y='Y', hue='Iteration', linewidth=0, legend=False)
    plot.set_xlabel('')
    plot.set_ylabel('')
    plot.set_xticklabels([])
    plot.set_yticklabels([])
    
    plt.show()
    return 0

if __name__ == "__main__" :
    main()  