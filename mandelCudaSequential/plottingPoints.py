#import pip

#pip.main['install', 'seaborn']

# import seaborn
import pandas
import matplotlib.pyplot as plt

def main() :

    # Process input
    csv = pandas.read_csv("C:\\Users\\Devil\\Desktop\\Random Docs\\CudaProgramming\\mandelCuda\\mandelCudaSequential\\MandelSetOut.txt")
    # seaborn.scatterplot(csv)
    plt.show()
    return 0

if __name__ == "__main__" :
    main()  