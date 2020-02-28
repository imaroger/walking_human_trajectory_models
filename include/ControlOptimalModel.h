#ifndef CONTROLOPTIMALMODEL_H 
#define CONTROLOPTIMALMODEL_H 

#include "crocoddyl/core/actions/human.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace crocoddyl;

class ControlOptimalModel
{
private :

	Eigen::Vector3d final_state;
	Eigen::Vector3d init_state;
	Eigen::Vector3d final_state_translated;
	Eigen::VectorXd init_state_translated = Eigen::VectorXd(6);	
	Eigen::VectorXd cost_weights = Eigen::VectorXd(7);

	int T_guess;
	double alpha;
	vector<boost::shared_ptr<ActionModelAbstract>> running_models;
	ActionModelHuman model;

public : 

	int T_opt;
	vector<double> x;
	vector<double> y;
	vector<double> theta;	

	ControlOptimalModel(double x0, double y0, double theta0, 
		double xf, double yf, double thetaf,
		Eigen::VectorXd costs,double a);

	void optimizeT(boost::shared_ptr<ActionModelAbstract> model_ptr);

	void solve();
};

#endif