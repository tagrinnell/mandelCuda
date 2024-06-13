#
#   Plotting Mandelbrot points from a csv with x,y, and iteration #
#   Iteration # refers to the iteration that a point diverges
#
#   Author: Tasman Grinnell
#

import pandas
from PIL import Image

# Colors:
# Black         - rgb(0, 0, 0) ~~ it. 1000
# Light brown   - rgb(255, 204, 153) ~~ 500
# Fuschia       - rgb(255, 0, 255) ~~ 1


def main() :
    # Process input
    # Laptop
    # csv = pandas.read_csv("C:\\Users\\Devil\\Desktop\\Random Docs\\CudaProgramming\\mandelCuda\\mandelCudaSequential\\MandelSetOut.txt")
    
    # PC
    csv = pandas.read_csv("C:\\Users\\tasma\\Desktop\\Textbooks\\mandelCuda\\mandelCudaSequential\\MandelSetOut.csv")

    # PIL.Image.new(mode: str, size: tuple[int, int], color: float | tuple[float, ...] | str | None = 0) → Image[source]
    # use mode = RGB

    imageSize = [csv.sizeX(0), csv.sizeY(0)]

    newIm = Image.new("RGB", )

    return 0


if __name__ == "__main__" :
    main()  