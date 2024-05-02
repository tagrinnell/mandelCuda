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

// using namespace std;

#define X 128
#define Y 72

struct Point {
    int x;
    int y;
    int iteration;
};

int main () {

    std::vector <struct Point> mandelSet;

    unoptimized_Escape(&mandelSet);

    // Use GNUPlot somehow to plot this


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