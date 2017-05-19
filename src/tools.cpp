#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Declare RMSE vector and set to (0,0,0,0)
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   // Check if estimations size is above 0 and if
   // the estimations vector is the same size as the
   // ground truth vector
   if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
     cout << "Invalid estimation or ground truth vector size" << endl;
     return rmse;
   }


   // Sum squared residuals
   for (int i = 0; i < estimations.size(); i += 1) {
     VectorXd residual = estimations[i] - ground_truth[i];
     residual = residual.array() * residual.array();
     rmse += residual;
   }

   // Calculate the mean
   rmse = rmse / estimations.size();
   // Calculate the square root
   rmse = rmse.array().sqrt();

   // Return the RMSE;
   return rmse;
}
