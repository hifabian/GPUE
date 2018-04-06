/*
* overdosing.hpp - GPUE2: GPU Split Operator solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
*/

#ifndef OVER_H
#define OVER_H

double2 operator+(double2 const& d1, double2 const& d2);
double2 operator+(double2 const& d1, double2 const& d2){
    double2 result;
    result.x = d1.x + d2.x;
    result.y = d1.y + d2.y;
    return result;
}

double2 operator*(double2 const& d1, double2 const& d2);
double2 operator*(double2 const& d1, double2 const& d2){
    double2 result;
    result.x = (d1.x*d2.x - d1.y*d2.y);
    result.y = (d1.x*d2.y + d1.y*d2.x);
    return result;
}

double2 operator*(double const& d1, double2 const& d2);
double2 operator*(double const& d1, double2 const& d2){
    double2 result;
    result.x = d1*cmp.x;
    result.y = d1*cmp.y;
    return result;
}

#endif
