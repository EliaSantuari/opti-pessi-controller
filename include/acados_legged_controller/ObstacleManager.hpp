#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>


#include "acados_legged_controller/LipmConfig.hpp"




namespace legged {

// What OCP needs to know about obstacles
struct ObstacleState {
    double x, y;            // m
    double radius;          // m
    double y_dot_max;       // m/s
    double a_max = 1.0;     // m/s^2
    double vx = 0.0;        // m/s
    double vy = 0.0;        // m/s
    double e_v = 0.0;       // m/s, velocity unceratinty (>= 2 y_dot_max if velocity unknown: same margin as Leonardelli's)
    double t0 = 0.0;        // s, age of measurement 
};




// Support function
inline double supportMargin(double u0, double t, double v_max, double a_max){
    u0 = std::clamp(u0, -v_max, v_max);
    a_max = std::max(a_max, 1e-3);
    const double t_star = (v_max - u0) / a_max;
    const double rest = std::max(t_star - t, 0.0);
    return v_max * t - 0.5 * a_max * t_star * t_star + 0.5 * a_max * rest * rest;
}


// Obstacle margin beyond radius for the line with normal (ax, ay)
inline double obstacleMargin(const std::string& kind, double ax, double ay, const ObstacleState& o, double t){
    const double u0 = -(ax * o.vx + ay * o.vy) + o.e_v;
    if (kind == "leonardelli") return o.y_dot_max * t;
    if (kind == "min_max") return std::min(o.y_dot_max * t, u0 * t + 0.5 * o.a_max * t * t);
    return supportMargin(u0, t, o.y_dot_max, o.a_max);
}



// Base class: each subclass defines a behaviour in update()
class Obstacle {

public:
    explicit Obstacle(const ObstacleState& state) : state_(state) {}
    virtual ~Obstacle() = default;
    
    virtual void update(double dt, double t, double robot_x, double robot_y) = 0;
    // true: it follows the robot, so it stands still while the robot is not walking (updateAll)
    virtual bool chasesRobot() const { return false; }
    const ObstacleState& state() const { return state_; }

protected:
    ObstacleState state_;
};



// STATIC
class StaticObstacle : public Obstacle {
public:
    using Obstacle::Obstacle;
    void update(double, double, double, double) override {}
};





// DYNAMIC BACK AND FORTH: trapezoidal speed profile (|acceleration| <= a_max), stops at the two ends
class DynamicObstacle : public Obstacle {
public:
    DynamicObstacle(const ObstacleState& state, double end_x, double end_y, double speed)
        : Obstacle(state), start_x_(state.x), start_y_(state.y), end_x_(end_x), end_y_(end_y),
          length_(std::hypot(end_x - state.x, end_y - state.y)), acc_(std::max(state.a_max, 1e-3)) {
        // triangular profile if the path is too short to reach the speed
        v_peak_ = std::min(speed, std::sqrt(acc_ * length_));
        t_acc_ = v_peak_ / acc_;
        t_leg_ = (v_peak_ > 1e-9) ? 2.0 * t_acc_ + (length_ - v_peak_ * t_acc_) / v_peak_ : 0.0;
    }

    void update(double, double t, double, double) override {
        if (length_ < 1e-5 || t_leg_ <= 0.0) return;
        double tau = std::fmod(t, 2.0 * t_leg_);
        const bool back = tau > t_leg_;     // second half: way back
        if (back) tau -= t_leg_;

        double s, v;    // distance and speed along the leg
        if (tau < t_acc_) {
            s = 0.5 * acc_ * tau * tau;
            v = acc_ * tau;
        } else if (tau < t_leg_ - t_acc_) {
            s = 0.5 * v_peak_ * t_acc_ + v_peak_ * (tau - t_acc_);
            v = v_peak_;
        } else {
            const double rest = t_leg_ - tau;
            s = length_ - 0.5 * acc_ * rest * rest;
            v = acc_ * rest;
        }
        if (back) { s = length_ - s; v = -v; }

        const double ux = (end_x_ - start_x_) / length_;
        const double uy = (end_y_ - start_y_) / length_;
        state_.x = start_x_ + s * ux;
        state_.y = start_y_ + s * uy;
        state_.vx = v * ux;
        state_.vy = v * uy;
    }

private:
    double start_x_, start_y_, end_x_, end_y_;
    double length_;
    double acc_;
    double v_peak_ = 0.0, t_acc_ = 0.0, t_leg_ = 0.0;
};





// CIRCULAR: around a centre, starting from pos, constant speed along the circle (radius = |pos - centre|).
// Position from the time of the motion like DYNAMIC: same path at every test, restarts from pos after a reset
class CircularObstacle : public Obstacle {
public:
    CircularObstacle(const ObstacleState& state, double center_x, double center_y, double speed, bool clockwise)
        : Obstacle(state), center_x_(center_x), center_y_(center_y),
          radius_(std::hypot(state.x - center_x, state.y - center_y)),
          angle0_(std::atan2(state.y - center_y, state.x - center_x)),
          omega_((clockwise ? -1.0 : 1.0) * (radius_ > 1e-5 ? speed / radius_ : 0.0)) {
            if (radius_ > 1e-5 && speed * speed / radius_ > state.a_max) {
                throw std::runtime_error("[ObstacleManager]: circular obstacle with speed^2/radius > a_max");
            }
            state_.vx = -omega_ * radius_ * std::sin(angle0_);
            state_.vy = omega_ * radius_ * std::cos(angle0_);
          }

    void update(double, double t, double, double) override {
        const double angle = angle0_ + omega_ * t;
        state_.x = center_x_ + radius_ * std::cos(angle);
        state_.y = center_y_ + radius_ * std::sin(angle);
        state_.vx = -omega_ * radius_ * std::sin(angle);
        state_.vy = omega_ * radius_ * std::cos(angle);
    }

private:
    double center_x_, center_y_;
    double radius_;
    double angle0_;
    double omega_;  // rad/s, negative = clockwise
};



// ADVERARIAL: straight towards the robot
class AdversarialObstacle : public Obstacle {
public: 
    AdversarialObstacle(const ObstacleState& state, double speed) : Obstacle(state), speed_(speed) {}
    bool chasesRobot() const override { return true; }


    void update(double dt, double, double robot_x, double robot_y) override {
        if (dt <= 0.0) { state_.vx = 0.0; state_.vy = 0.0; return; }   // robot not walking: stands still
        const double dir_x = robot_x - state_.x;
        const double dir_y = robot_y - state_.y;
        const double dist = std::hypot(dir_x, dir_y);
        const double vdx = (dist > 1e-3) ? dir_x / dist * speed_ : 0.0;
        const double vdy = (dist > 1e-3) ? dir_y / dist * speed_ : 0.0;
        // |dv| <= a_max dt
        double dvx = vdx - state_.vx, dvy = vdy - state_.vy;
        const double dv = std::hypot(dvx, dvy), dv_max = state_.a_max * dt;
        if (dv > dv_max) { dvx *= dv_max / dv; dvy *= dv_max / dv; }
        state_.vx += dvx;
        state_.vy += dvy;
        state_.x += state_.vx * dt;
        state_.y += state_.vy * dt;
    }


private:
    double speed_;

};


// RANDOM: constant speed, noisy heading, bounces on borders of an area
class RandomObstacle : public Obstacle {
public:
    RandomObstacle(const ObstacleState& state, double speed, const std::array<double, 4>& bounds, unsigned int seed)
        : Obstacle(state), speed_(speed), bounds_(bounds), rng_(seed), noise_(-1.0, 1.0) {
            heading_ = M_PI * noise_(rng_);
            state_.vx = speed_ * std::cos(heading_);
            state_.vy = speed_ * std::sin(heading_);
        }


    void update(double dt, double, double, double) override {
        if (dt <= 0.0) return;
        const double acc = std::max(state_.a_max, 1e-3);
        const double turn_max = acc * dt / std::max(speed_, 1e-3);   // speed * dheading/dt <= a_max
        const double turn_radius = speed_ * speed_ / acc;
        // sqrt(dt): heading noise per second does not depend on update rate
        double turn = noise_(rng_) * std::sqrt(0.175 * dt);

        // close to the border and going out: turn back towards the centre of the area
        const bool near_border = state_.x < bounds_[0] + turn_radius || state_.x > bounds_[1] - turn_radius ||
                                 state_.y < bounds_[2] + turn_radius || state_.y > bounds_[3] - turn_radius;
        if (near_border) {
            const double to_centre = std::atan2(0.5 * (bounds_[2] + bounds_[3]) - state_.y,
                                                0.5 * (bounds_[0] + bounds_[1]) - state_.x);
            const double err = std::remainder(to_centre - heading_, 2.0 * M_PI);
            if (std::abs(err) > M_PI_2) turn = err;
        }
        heading_ += std::clamp(turn, -turn_max, turn_max);

        state_.vx = speed_ * std::cos(heading_);
        state_.vy = speed_ * std::sin(heading_);
        state_.x += state_.vx * dt;
        state_.y += state_.vy * dt;
    }


private:
    double speed_;
    std::array<double, 4> bounds_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> noise_;
    double heading_;

};




class ObstacleManager {
public:
    explicit ObstacleManager(const LipmConfig& cfg) : n_obs_max_(cfg.N_OBS_MAX) {
        const auto& rnd = cfg.random_obstacles;
        if (static_cast<int>(cfg.obstacles.size()) + rnd.count > n_obs_max_){
            throw std::runtime_error("[ObstacleManager]: more obstacle than N_OBS_MAX!!");
        }
        for (const auto& obs : cfg.obstacles){
            ObstacleState state{obs.x, obs.y, obs.r_obs, obs.y_dot_max, obs.a_max};
            state.e_v = cfg.vel_uncertainty;
            // WE WILL ADD NEW BEHAVIOURS HERE
            if (obs.type == "static") {
                obstacles_.push_back(std::make_unique<StaticObstacle>(state));
            } else if (obs.type == "dynamic") {
                obstacles_.push_back(std::make_unique<DynamicObstacle>(state, obs.end_x, obs.end_y, obs.speed));
            } else if (obs.type == "circular") {
                obstacles_.push_back(std::make_unique<CircularObstacle>(state, obs.center_x, obs.center_y, obs.speed, obs.clockwise));
            } else if (obs.type == "adversarial") {
                obstacles_.push_back(std::make_unique<AdversarialObstacle>(state, obs.speed));
            } else {
                throw std::runtime_error("[ObstacleManager]: unknown obstacle type '" + obs.type + "'");
            }
        }
        // Random obstacle: position, radius and speed drawn from ranges in config.yaml
        std::mt19937 rng(rnd.seed);
        auto uniform = [&rng](double min, double max) {
            return std::uniform_real_distribution<double>(min, max)(rng);
        };
        for (int i = 0; i < rnd.count; ++i){
            ObstacleState state{
                uniform(rnd.bounds[0], rnd.bounds[1]), uniform(rnd.bounds[2], rnd.bounds[3]),
                uniform(rnd.r_range[0], rnd.r_range[1]), rnd.y_dot_max, rnd.a_max
            };
            state.e_v = cfg.vel_uncertainty;
            const double speed = uniform(rnd.speed_range[0], rnd.speed_range[1]);
            obstacles_.push_back(std::make_unique<RandomObstacle>(state, speed, rnd.bounds, rng()));
        }
    }


    // robot_walking false: the obstacles that chase the robot (adversarial) stand still, the others keep moving
    void updateAll(double dt, double t, double robot_x, double robot_y, bool robot_walking = true){
        for (const auto& obs : obstacles_) {
            obs->update((robot_walking || !obs->chasesRobot()) ? dt : 0.0, t, robot_x, robot_y);
        }
    }


    // Only real obstacles
    std::vector<ObstacleState> active() const{
        std::vector<ObstacleState> states;
        for (const auto& obs : obstacles_) states.push_back(obs->state());
        return states;
    }


    // For all N_OBS_MAX obstacles: free slots are far-away ghosts
    std::vector<ObstacleState> slots() const {
        std::vector<ObstacleState> states = active();
        for (int i = static_cast<int>(states.size()); i < n_obs_max_; ++i){
            states.push_back({kGhostObsPos + 2.0 * i, kGhostObsPos, 0.0, 0.0});
        }
        return states;
    }


private:
    static constexpr double kGhostObsPos = 15;
    int n_obs_max_;
    std::vector<std::unique_ptr<Obstacle>> obstacles_;


};



} // namespace legged