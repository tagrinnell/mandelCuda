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
#include <chrono>

#define X 19200.0
#define Y 10800.0

long int numIterations;

struct Point {
    int x;
    int y;
    int iteration;
};

void unoptimized_Escape(std::vector<struct Point> *set);

int main () {

    std::vector <struct Point> mandelSet;

    numIterations = 0;

    auto tStart = std::chrono::high_resolution_clock::now();

    unoptimized_Escape(&mandelSet);

    auto tEnd = std::chrono::high_resolution_clock::now() ;

    // Write data out to a file
    std::ofstream file ("MandelSetOut.csv");
    std::ofstream timing ("timing_unopt.txt");

    // fprintf(file, "# X\tY");
    file << "X,Y,Iteration" << std::endl;

    for (int i = 0; i < mandelSet.size(); i++) {
        file << mandelSet.at(i).x << "," << mandelSet.at(i).y << "," << mandelSet.at(i).iteration << std::endl;
    }

    file.close();

    const std::chrono::duration<double, std::milli> dur = (tEnd - tStart);

    timing << numIterations << " Iterations in " << dur.count() << "ms using params:\n X = " << X << ", Y = "<< Y<< std::endl;

    timing.close();

    return 0;
}

/*
 * Mandelbrot Calculation using unoptimized Escape 
**/
void unoptimized_Escape(std::vector<struct Point> *set) {
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {

            float x0 = i / X * 2.47 - 2;
            float y0 = j / Y * 2.24- 1.12;
            float x = 0.0;
            float y = 0.0;

            int iteration = 0;
            int max_iteration = 1000;

            while (x * x + y * y <= (2 * 2) && iteration < max_iteration) {
                float xtemp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = xtemp;
                
                iteration++;
            }

            struct Point newPoint = {i, j, iteration};
            set->push_back(newPoint);
            
            numIterations++;
        }

        numIterations++;
    }
}