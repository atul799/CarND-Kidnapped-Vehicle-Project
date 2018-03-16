/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
//random engine global declaration as it is used across multiple methods
default_random_engine gen;

//for dbg, keep a track of particle(out of 100), how many times re-sampled

//vector<int> particle_sample_count(num_particles,0);


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//Set number of particles
	//num_particles=1000;
	//num_particles=100;
	num_particles=2;

	//random engine declaration
	//default_random_engine gen;



	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	//Set standard deviations for x, y, and theta.
	std_x=std[0];
	std_y=std[1];
	std_theta=std[2];

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std_x);
	//Create normal distributions for y and theta.
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);


	for (int j=0;j < num_particles;j++) {
		Particle p;
		p.id=j;
		p.weight=1.0;
		// Sample  and from these normal distrubtions
		//sample_x = dist_x(gen);
		p.x=dist_x(gen);
		// where "gen" is the random engine initialized earlier.
		//sample_y = dist_y(gen);
		p.y=dist_y(gen);
		//sample_theta = dist_theta(gen);
		p.theta=dist_theta(gen);
		// Print your samples to the terminal.
		//cout << "Sample " <<  p.x << " " << p.y << " " << p.theta << endl;
		p.nr_times_resampled=0;


		particles.push_back(p);
		//particle_cloud[j]=0;
		cout<<"particle number: " << j << " initialized to: id:" << p.id<<" x: " <<p.x<<" y: " << p.y<< " theta: "<<p.theta<<" weight: "<<p.weight<<endl;
	}

	//set particle initialized flag to true
	is_initialized = true;


	cout <<"Particles initialized"<<endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	//from UKF motion model, px and py depend on yawrate in denominator
	//hence check for too small a yawrate and update px,py accordingly
	//fabs(yawrate) >= 0.001
	//xf=xinit+ (vel/yawrate)*(sin(yawinit+yawrate*dt)-sin(yawinit))
	//yf=yinit+ (vel/yawrate)*(cos(yawinit)-cos(yawinit+yawrate*dt))

	//fabs(yawrate) <0.001
	//xf=xinit+vel*dt*cos(yawinit)
	//yf=yinit+vel*dt*sin(yawinit)


	//thetaf=thetainit + yawrate*dt

	//add noise
	//xf=normal_distribution<double> dist_x(xf, std_x);
	//yf=normal_distribution<double> dist_y(yf, std_y);
	//thetaf=normal_distribution<double> dist_theta(thetaf, std_theta);

	// define normal distributions for sensor noise
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	//Set standard deviations for x, y, and theta.
	std_x=std_pos[0];
	std_y=std_pos[1];
	std_theta=std_pos[2];

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(0, std_x);
	//Create normal distributions for y and theta.
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);
	// predict new state for each particle
	for (int i = 0; i < num_particles; i++) {


		if (fabs(yaw_rate) < 0.001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
			//no change in theta as yaw_rate is  effectively 0
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);

		//cout <<"Particle: " << i <<" moved to: " <<" x: " <<particles[i].x<<" y: " <<particles[i].y<<" theta: " <<particles[i].theta<<endl;
	}




}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//  Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	//for each observation calculate eucladean distance to each predicted (landmark)
	for (int i = 0; i < observations.size(); i++) {

		// local var for current observation (struct LandmarkObs used)
		LandmarkObs obs = observations[i];

		// keep a counter of min distance to each landmark
		//assuming map size < 1000 and that would be max distance
		double min_distance = 1000;

		//id of landmark to associate to observation[i]
		int landmark_id ;

		for (int j = 0; j < predicted.size(); j++) {
			// current prediction
			LandmarkObs pred = predicted[j];

			//dist between current obs and predicted landmarks
			//function dist is in helper_functions.h file
			double run_dist = dist(obs.x, obs.y, pred.x, pred.y);

			//update nearest neighbor
			// find the predicted landmark nearest the current observed landmark
			if (run_dist < min_distance) {
				min_distance = run_dist;
				landmark_id = pred.id;
			}
		}

		//observation[i] id to the nearest neighbor
		observations[i].id = landmark_id;
		//cout << "observation: " << i <<" is assigned to landmark: "<< landmark_id<<endl;
	}

	//can return landmark_id from this method and add it to particle[i].associations
	//that will avoid searching through all landmarks to find associated landmark

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//  Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	/////////////////////////////////////////////
	//for each particle:
		//Step 0. pick landmarks that are in sensor_range range to particle, landmarks and particles are in map
		//co-ordinates
		//Step 1. transform each of observations to map co-ordinate w.r.t. each particle
		//xm=xp+cos(theta)*xo-sin(theta)*yo
		//ym=yp+sin(theta)*xo+cos(theta)*yo

		//Step 2. find associations for each particle to landmarks (sense)

		//Step 3. find measurement probabilities and update weights
	///////////////////////////////////////////////

	// calculate constants for multivariate Gaussian PD
	const double norm_f = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	//denoms of x and y term in exponentials of PD
	const double x_d = 2 * std_landmark[0] * std_landmark[0];
	const double y_d = 2 * std_landmark[1] * std_landmark[1];




	//loop over all particles
	for (int i=0;i<num_particles;i++) {


		//Step 0
		//pick landmarks within sensor_range and store them to only process those for given particle
		vector<LandmarkObs> landmarks_in_range;
		for (int j=0; j < map_landmarks.landmark_list.size();j++) {
			//check distance to particle
			//double dist_x=map_landmarks.landmark_list[j].x_f-particles[i].x;
			//double dist_y=map_landmarks.landmark_list[j].y_f-particles[i].y;
			//double dist_euclead= sqrt(pow(dist_x,2)+pow(dist_y,2));
			double dist_euclead= dist(map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f,particles[i].x,particles[i].y);
			//if (fabs(dist_x) <= sensor_range && fabs(dist_y) <= sensor_range) {
			if (dist_euclead < sensor_range) {
				//push land marks in range to landmarks_in_range
				landmarks_in_range.push_back({map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f});
			}
		}
		cout << "for particle :" << i << " number of landmarks associated are: " << landmarks_in_range.size()<<endl;
		//Step 1
		//transform each observation to map cordinates w.r.t particle
		vector<LandmarkObs> observations_in_map_cord;
		for (int k=0;k< observations.size();k++) {
			//
			double x_transform=particles[i].x + cos(particles[i].theta)*observations[k].x-sin(particles[i].theta)*observations[k].y;
			double y_transform=particles[i].y + sin(particles[i].theta)*observations[k].x+cos(particles[i].theta)*observations[k].y;
			//push back observation id, transformed x and y to observations_in_map_cord
			observations_in_map_cord.push_back(LandmarkObs{observations[k].id,x_transform,y_transform});
		}

		cout << "for particle :" << i << " number of observations in map cords are: " << observations_in_map_cord.size()<<endl;
		//Step 2. find associations for each particle to landmarks (sense)
		//landmarks_in_range passed as value and observations_in_map_cord passed as reference to method dataAssociation
		dataAssociation(landmarks_in_range,observations_in_map_cord );

		cout << "after data association" <<endl;
		for (int pp=0;pp<observations_in_map_cord.size();pp++) {
			cout <<"Obs: " << pp << " associated to landmark id: " << observations_in_map_cord[pp].id<<endl;
			cout <<"base Observation x: " << observations[pp].x<< " base observation y: " << observations[pp].y<<endl;
			cout <<"particle pos  x: " << particles[i].x << " particle pos y: " << particles[i].y<< " particle pos theta: " << particles[i].theta<<endl;
			cout <<"transformed Observation x: " << observations_in_map_cord[pp].x<< " Observation y: " << observations_in_map_cord[pp].y<<endl;
			//landmark position for associated observation
		}

		//Step 3. find measurement probabilities and update weights
		//A> for each transformed observation find landmark associated with each (search with id)
			//making a hash table would probably have made it efficient to search for id
		//B> calculate multivariate PD for each observation and multiply by previous weights for given particle

		for (int l=0;l<observations_in_map_cord.size();l++) {

			//id of landmark associated with the current observation
			int assoc_lm=observations_in_map_cord[l].id;
			//cout << "for particle : "<< i << " observation assigned to landmark: " << assoc_lm<<endl;
			//variable to capture x,y of  landmark associated with observation
			double x_lm_p,y_lm_p;

			//loop through landmarks_in_range and find the one with same id as
			//assoc_lm

			for (int m=0;m < landmarks_in_range.size();m++) {
				if (assoc_lm==landmarks_in_range[m].id) {
					x_lm_p=landmarks_in_range[m].x;
					y_lm_p=landmarks_in_range[m].y;
					//break;
				}
			}
			//cout << "landmark position x: "<< x_lm_p << "landmark position y: " << y_lm_p<<endl;
			//cout << "observation position x: "<< observations_in_map_cord[l].x << "observation position y:" << observations_in_map_cord[l].y<<endl;
			//calculate multivariate PD for the w.r.t. associated landmark
			double l_x_diff=observations_in_map_cord[l].x-x_lm_p;
			double l_y_diff=observations_in_map_cord[l].y-y_lm_p;
			//cout << "local x diff: " << l_x_diff << " local y diff : "<<l_y_diff<<endl;
			double x_diff_sq=pow(l_x_diff,2);
			double y_diff_sq=pow(l_y_diff,2);
			//cout << "local x diff sq : " << x_diff_sq << " local y diff sq : "<<y_diff_sq<<endl;
			double mv_G=norm_f*exp(-(x_diff_sq/x_d + y_diff_sq/y_d));
			//cout << "multivariate gaussian for observation: " << mv_G<<endl;
			// multiple with mvgaussian pdf with prior weights
			particles[i].weight *= mv_G;
			cout << "particle number: "<<i << "now has weight: "<<particles[i].weight<<endl;


		}



	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	/////////////
	//sebastian code on importance sampling (sampling with replacement)
	/* p3 = [] //new set of particles will be saved as original particle of IS
	        index = int(random.random() * N)
	        beta = 0.0
	        mw = max(w)
	        for i in range(N):
	            beta += random.random() * 2.0 * mw
	            while beta > w[index]:
	                beta -= w[index]
	                index = (index + 1) % N
	            p3.append(p[index])
	        p = p3 */

	//Step 1: create a new vector of type Partcile and  size num_partciles and a weight vector
	//Step 2: generate a random index from num_particles range
	//	create a variable with max weights of all particles
	//Step 3: for each particle:
	//   generate a random number beta-->between 0 and 2*maxweights
	// 	if beta > weight of current indexed particles (small weight partcile)
	//	subtract the beta contribution from the weights of particle
	//	increment the index of particle
	//	else choose the partcile at current index

	///////



	//Step 1
	vector<Particle> p_n;
	vector<double> weights;

	//captures weight of each particle in weights vector
	for (int i=0;i<num_particles;i++) {
		weights.push_back(particles[i].weight);
	}

	//Step 2
	//use of discrete_distribution to resample
	discrete_distribution<> d(weights.begin(),weights.end());

	//particles that survived
	vector<int> particles_survived;
	for (int j=0;j<num_particles;j++) {
		//get index according to discrete distribtuion sampling
		int idx=d(gen);
		//cout << "Particle: " << idx << " got sampled" << endl;
		//++particles[idx].nr_times_resampled;
		//cout << "nr of times particle: " << idx << " sampled so far: " <<  particles[idx].nr_times_resampled << endl;
		p_n.push_back(particles[idx]);
		particles_survived.push_back(idx);

	}




	// reset particles
	particles=p_n;






}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}



