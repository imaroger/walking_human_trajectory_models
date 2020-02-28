#include "OptimalControlModel.h"
#include <fstream>

using namespace std;
using namespace crocoddyl;

OptimalControlModel::OptimalControlModel(double x0, double y0, double theta0, 
		double xf, double yf, double thetaf, Eigen::VectorXd costs,double a)
{
	final_state << xf,yf,thetaf;
	init_state << x0,y0,theta0;
	cout << "init (R) : " << init_state[0] << " | " << init_state[1] << " | " << init_state[2] << endl; 
	cout << "final (R) : " << final_state[0] << " | " << final_state[1] << " | " << final_state[2] << endl; 	
	final_state_translated << (xf-x0)*cos(theta0)+(yf-y0)*sin(theta0),
	-(xf-x0)*sin(theta0)+(yf-y0)*cos(theta0),thetaf-theta0;
	cout << "final (0) : " << final_state_translated[0] << " | " << final_state_translated[1] << " | " << final_state_translated[2] << endl; 
	init_state_translated << 0.,0.,0.,0.,0.,0.;
	cost_weights << costs[0],costs[1],costs[2],costs[3],costs[4],costs[5],costs[6];
	alpha = a;
	double distance;
	distance = sqrt(pow(final_state_translated[0],2)+pow(final_state_translated[1],2));
	T_guess = distance*100/alpha;
}

void OptimalControlModel::optimizeT(boost::shared_ptr<ActionModelAbstract> model_ptr)
{
	vector<boost::shared_ptr<ActionModelAbstract>> running_models;

	int T_min = T_guess;
	for(int i=0; i<T_min; ++i)
	{
		running_models.push_back(model_ptr);
	}		
	int T_max = T_guess + 150;
	T_opt = T_min;
	double cost_min = -1;
	double cost;
	for (int t = T_min; t <= T_max; t += 5)
	{
		ShootingProblem problem(init_state_translated,running_models,model_ptr);

		boost::shared_ptr<ShootingProblem> problem_ptr;
		problem_ptr = boost::make_shared<ShootingProblem>(problem);	
		SolverDDP ddp(problem_ptr);
		bool done = ddp.solve();
		vector<Eigen::VectorXd> xs = ddp.get_xs();
		vector<Eigen::VectorXd> us = ddp.get_us();			
		cost = cost_weights[0] + cost_weights[1]*pow(us[us.size()-1][0],2)
			+ cost_weights[2]*pow(us[us.size()-1][1],2)
			+ cost_weights[3]*pow(us[us.size()-1][2],2)
			+ cost_weights[4]*pow(atan2(final_state_translated[1]-xs[xs.size()-1][1]
			,final_state_translated[0]-xs[xs.size()-1][0])-xs[xs.size()-1][2],2)
			+cost_weights[5]*(pow(xs[xs.size()-1][0]-final_state_translated[0],2)
			+pow(xs[xs.size()-1][1]-final_state_translated[1],2))
			+cost_weights[6]*pow(xs[xs.size()-1][2]-final_state_translated[2],2);
		if(cost_min < 0 || cost_min > cost && ddp.get_iter() < 80)
		{
			cost_min = cost;
			T_opt = t;
		}
		for(int i=0; i<5; ++i)
		{
			running_models.push_back(model_ptr);
		}
	}
}

void OptimalControlModel::solve()
{
	model.set_cost_weights(cost_weights);
	model.set_final_state(final_state_translated);	
	model.set_alpha(alpha);
	boost::shared_ptr<ActionModelAbstract> model_ptr;
	model_ptr = boost::make_shared<ActionModelHuman>(model);

	optimizeT(model_ptr);

	vector<boost::shared_ptr<ActionModelAbstract>> running_models;
	for(int i=0; i<T_opt; ++i)
	{
		running_models.push_back(model_ptr);
	}	
	ShootingProblem problem(init_state_translated,running_models,model_ptr);
	boost::shared_ptr<ShootingProblem> problem_ptr;
	problem_ptr = boost::make_shared<ShootingProblem>(problem);	

	SolverDDP ddp(problem_ptr);
	ddp.solve();
	vector<Eigen::VectorXd> xs;
	xs = ddp.get_xs();

	for (int i = 0; i < xs.size(); i++)
	{
		x.push_back(init_state[0]+xs[i][0]*cos(init_state[2])-xs[i][1]*sin(init_state[2]));
		y.push_back(init_state[1]+xs[i][0]*sin(init_state[2])+xs[i][1]*cos(init_state[2]));		
		theta.push_back(xs[i][2]+init_state[2]);
	}
}


int main(int argc, char **argv)
{	
	Eigen::VectorXd cost_weights = Eigen::VectorXd(7);
	cost_weights << 1., 1.2, 1.7, 0.7, 5.2, 5. , 8.;
	double alpha = 0.1;
	OptimalControlModel co(-0.32,1.51,0.,1.6,1.74,0.26,cost_weights,alpha);

	clock_t t;
	t = clock();
	printf ("Calculating...\n");

	co.solve();

	t = clock() - t;
	printf ("It took %f seconds.\n",((float)t)/CLOCKS_PER_SEC);

	cout << co.x.size() << endl; 
	cout << co.x[0] << " | " << co.y[0] << " | " << co.theta[0]  << endl;
	cout << co.x[co.x.size()-1] << " | " << co.y[co.y.size()-1] << " | " << co.theta[co.theta.size()-1]  << endl;

	string filename_x = "/local/imaroger/catkin_ws/src/trajectory_generation/data/OptControl/x.dat";
	string filename_y = "/local/imaroger/catkin_ws/src/trajectory_generation/data/OptControl/y.dat";
	string filename_th = "/local/imaroger/catkin_ws/src/trajectory_generation/data/OptControl/theta.dat";
	
	std::ofstream file_x(filename_x);
	std::ofstream file_y(filename_y);
	std::ofstream file_th(filename_th);	

	for (int i = 0; i<co.x.size(); i++)
	{
	    file_x << to_string(co.x[i]);
	    file_x << "\n";

	    file_y << to_string(co.y[i]);
	    file_y << "\n";		    
	    
	    file_th << to_string(co.theta[i]);
	    file_th << "\n";
	}

	file_x.close();	
	file_y.close();	
	file_th.close();	
}
