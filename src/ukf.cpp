#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.399;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  x_ << 1, 1, 1, 1, 1;

  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  // INITIALIZE WEIGHTS
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  MatrixXd Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0,
        0, std_yawdd_ * std_yawdd_;
}

UKF::~UKF() {}

void UKF::NormalizeAngle(double& phi)
{
  phi = atan2(sin(phi), cos(phi));
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    x_.fill(0.0);
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float range = meas_package.raw_measurements_(0);
      float bearing = meas_package.raw_measurements_(1);
      float radial_velocity = meas_package.raw_measurements_(2);
      // Transform to Cartesian coordinates
      const float px = range * cos(bearing);
      const float py = range * sin(bearing);
      const float vx = radial_velocity * cos(bearing);
      const float vy = radial_velocity * sin(bearing);
      const float v = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, 0, 0;
    }
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      if (fabs(x_(0)) < 0.001 and fabs(x_(1)) < 0.001) {
        x_(0) = 0.001;
        x_(1) = 0.001;
      }
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

 // Augmentation
 VectorXd x_aug = VectorXd(n_aug_);
 MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
 MatrixXd Xsig_aug_ = MatrixXd(n_aug_, (2 * n_aug_ + 1));

 x_aug.fill(0.0);
 x_aug.head(n_x_) = x_;

 P_aug.fill(0.0);
 P_aug.topLeftCorner(n_x_, n_x_) = P_;
 P_aug(5, 5) = std_a_ * std_a_;
 P_aug(6, 6) = std_yawdd_ * std_yawdd_;

 MatrixXd A = P_aug.llt().matrixL();

 Xsig_aug_.col(0) = x_aug;
 double sqrt_lambda_n_aug_ = sqrt(lambda_ + n_aug_);
 for (int i = 0; i < n_aug_; i += 1) {
   VectorXd sqrt_lambda_n_aug_A_ = sqrt_lambda_n_aug_ * A.col(i);
   Xsig_aug_.col(i + 1) = x_aug + sqrt_lambda_n_aug_A_;
   Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt_lambda_n_aug_A_;
 }

 // Sigma point prediction
 for (int i = 0; i < 2 * n_aug_ + 1; i += 1) {
   const double px = Xsig_aug_(0, i);
   const double py = Xsig_aug_(1, i);
   const double v = Xsig_aug_(2, i);
   const double yaw = Xsig_aug_(3, i);
   const double yawd = Xsig_aug_(4, i);
   const double nu_a = Xsig_aug_(5, i);
   const double nu_yawdd = Xsig_aug_(6, i);

   double px_p, py_p;
   // AVOID DIVISION BY ZERO
   if (fabs(yawd) > 0.001) {
     px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
     py_p = py + v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
   } else {
     px_p = px + v * cos(yaw) * delta_t;
     py_p = py + v * sin(yaw) * delta_t;
   }

   double v_p = v;
   double yaw_p = yaw + yawd * delta_t;
   double yawd_p = yawd;

   // ADD NOISE
   const double delta_t_2 = delta_t * delta_t;
   px_p = px_p + .5 * delta_t_2 * cos(yaw) * nu_a;
   py_p = py_p + .5 * delta_t_2 * sin(yaw) * nu_a;
   v_p = v_p + delta_t * nu_a;
   yaw_p = yaw_p + .5 * delta_t_2 * nu_yawdd;
   yawd_p = yawd_p + delta_t * nu_yawdd;

   Xsig_pred_(0, i) = px_p;
   Xsig_pred_(1, i) = py_p;
   Xsig_pred_(2, i) = v_p;
   Xsig_pred_(3, i) = yaw_p;
   Xsig_pred_(4, i) = yawd_p;
 }

 // Predicted state mean
 x_ = Xsig_pred_ * weights_;

 // Predicted state covariance matrix
 P_.fill(0.0);
 for (int i = 0; i < 2 * n_aug_ + 1; i += 1) {
   VectorXd xdiff = Xsig_pred_.col(i) - x_;
   NormalizeAngle(xdiff(3));
   P_ = P_ + weights_(i) * xdiff * xdiff.transpose();
 }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  MatrixXd H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  MatrixXd R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_;

  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_pred = H_laser_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd K = P_ * Ht * S.inverse();

  // New estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;
  NIS_laser_ = z.transpose() * S.inverse() * z;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  // transform sigma points into measurement space
  int n_z_ = 3;
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i += 1) {
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yawn = Xsig_pred_(3, i);
    Zsig(0, i) = sqrt(px * px + py * py);
    if (fabs(px) < 0.001 and fabs(py) < 0.001) {
      Zsig(1, i) = atan2(0.001, 0.001);
      Zsig(2, i) = (0.001 * cos(yawn) * v + 0.001 * sin(yawn) * v) / Zsig(0, i);
    } else {
      Zsig(1, i) = atan2(py, px);
      Zsig(2, i) = (px * cos(yawn) * v + py * sin(yawn) * v) / Zsig(0, i);
    }
  }

  // Calculate mean predict measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred = Zsig * weights_;

  // Calculate measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i += 1) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add noise R
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R <<  std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i += 1) {
    VectorXd xdiff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(xdiff(3));

    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));
    Tc = Tc + weights_(i) * xdiff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;
  NormalizeAngle(z_diff(1));
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  NIS_radar_ = z.transpose() * S.inverse() * z;
}
