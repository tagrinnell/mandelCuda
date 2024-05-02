/*

    Author: Tasman Grinnell 

    Unoptimized mandelbrot set calculation.  

    Pseudocode:
    for each pixel (Px, Py) on the screen do
        xx := 0.0
        y := 0.0
        iteration := 0
        max_iteration := 1000
        while (x*x + y*y ≤ 2*2 AND iteration < max_iteration) do
            xtemp := x*x - y*y + x0
            y := 2*x*y + y0
            x := xtemp
            iteration := iteration + 1
    
    color := palette[iteration]
    plot(Px, Py, color)
*/

// Initial Testing: 1280x720 points
// Scaled

#include <vector>
#include <iostream>
#include <fstream>

#define X 1280
#define Y 720

struct Point {
    int x;
    int y;
    int iteration;
};

void unoptimized_Escape(std::vector<struct Point> *set);

int main () {

    std::vector <struct Point> mandelSet;

    unoptimized_Escape(&mandelSet);

    // Write data out to a file

    // FILE *file = _popen("MandelSetOut.txt", "w");
    std::ofstream file ("MandelSetOut.txt");

    // fprintf(file, "# X\tY");
    file << "# X\tY\tIteration" << std::endl;

    for (int i = 0; i < mandelSet.size(); i++) {
        // fprintf(file, "%d\t%d\n", mandelSet.at(i).x, mandelSet.at(i).y);
        file << mandelSet.at(i).x << "\t" << mandelSet.at(i).y << "\t" << mandelSet.at(i).iteration << std::endl;
    }

    // _pclose(file);   
    file.close();

    return 0;
}

/*
 * Mandelbrot Calculation using unoptimized Escape 
**/
void unoptimized_Escape(std::vector<struct Point> *set) {
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {

            float x0 = i / 2.47 - 2.0;
            float y0 = j / 2.24 - 1.12;
            float x = 0.0;
            float y = 0.0;

            int iteration = 0;
            int max_iteration = 1000;

            while (x * x + y * y <= 2 * 2 && iteration < max_iteration) {
                float xtemp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = xtemp;
                
                iteration++;
            }

            struct Point newPoint = {i, j, iteration};
            set->push_back(newPoint);

        }
    }
}