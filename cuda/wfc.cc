

class WaveFunction {
private:
	double2 *value;
	int dim;
	int gridSize[3];
public:
	double2* getValue();
	setValue(double2*);
	int getDim();
	int[] getGridSize();
	double2* getAngle();
}


class Operator {
private:
	double *value;
	int dim;
	int gridSize[3];
public:
	double2* getValue();
	setValue(double2*);
	int getDim();
	int[] getGridSize();
	double2* getAngle();
}
