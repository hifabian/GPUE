/*
* overdosing.hpp - GPUE2: GPU Split Operator solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2015, Lee J. O'Riordan.

* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 2.1 of the
* License, or (at your option) any later version. This library is
* distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
* License for more details. You should have received a copy of the GNU
* Lesser General Public License along with this library; if not, write
* to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
* Boston, MA 02111-1307 USA
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
