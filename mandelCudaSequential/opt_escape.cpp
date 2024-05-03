/*

    Author: Tasman Grinnell 

    x2:= 0
    y2:= 0

    while (x2 + y2 ≤ 4 and iteration < max_iteration) do
        y:= 2 * x * y + y0
        x:= x2 - y2 + x0
        x2:= x * x
        y2:= y * y
        iteration:= iteration + 1
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

void optimized_Escape(std::vector<struct Point> *set);

int main () {

    std::vector <struct Point> mandelSet;

    optimized_Escape(&mandelSet);

    // Write data out to a file
    std::ofstream file ("MandelSetOut_opt.csv");

    // fprintf(file, "# X\tY");
    file << "X,Y,Iteration" << std::endl;

    for (int i = 0; i < mandelSet.size(); i++) {
        file << mandelSet.at(i).x << "," << mandelSet.at(i).y << "," << mandelSet.at(i).iteration << std::endl;
    }

    file.close();

    return 0;
}

/*
 * Mandelbrot Calculation using optimized Escape
    x2:= 0
    y2:= 0

    while (x2 + y2 ≤ 4 and iteration < max_iteration) do
        y:= 2 * x * y + y0
        x:= x2 - y2 + x0
        x2:= x * x
        y2:= y * y
        iteration:= iteration + 1
**/
void optimized_Escape(std::vector<struct Point> *set) {
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            float x0 = i / 2.47 - 2.0;
            float y0 = j / 2.24 - 1.12;
            
            float x2 = 0;
            float y2 = 0;

            int iteration = 0;
            int max_iteration = 1000;

            while (x2 + y2 <= 4 && iteration < max_iteration) {
                float x = x2 - y2 + x0;
                float y = x * y * 2 + y0;
                x2 = x * x;
                y2 = y * y;

                iteration++;
            }

            struct Point newPoint = {i, j, iteration};
            set->push_back(newPoint);

        }
    }
}