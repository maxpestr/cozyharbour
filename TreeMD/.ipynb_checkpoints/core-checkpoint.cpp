#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

struct Particle {
    double x, y;
    double vx, vy;
    double m;
};

class Simulation {
public:
    Simulation(size_t N, double box_size = 1.0)
        : L(box_size), dt(0.005) {
        particles.resize(N);
        init_random();
    }

    void step() {
        // Compute accelerations
        std::vector<std::pair<double, double>> acc(particles.size(), {0, 0});
        const double G = 0.1;

        for (size_t i = 0; i < particles.size(); ++i) {
            for (size_t j = i + 1; j < particles.size(); ++j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double r2 = dx*dx + dy*dy + 1e-6;
                double r = std::sqrt(r2);
                double F = G * particles[i].m * particles[j].m / r2;

                double ax = F * dx / (r * particles[i].m);
                double ay = F * dy / (r * particles[i].m);
                acc[i].first  += ax;
                acc[i].second += ay;
                acc[j].first  -= ax;
                acc[j].second -= ay;
            }
        }

        // Verlet integration step
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].vx += acc[i].first * dt;
            particles[i].vy += acc[i].second * dt;
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;

            // Wall reflection
            if (particles[i].x < -L || particles[i].x > L) {
                particles[i].vx *= -1;
                particles[i].x = std::clamp(particles[i].x, -L, L);
            }
            if (particles[i].y < -L || particles[i].y > L) {
                particles[i].vy *= -1;
                particles[i].y = std::clamp(particles[i].y, -L, L);
            }
        }

        // Collisions (naive O(N²))
        for (size_t i = 0; i < particles.size(); ++i) {
            for (size_t j = i + 1; j < particles.size(); ++j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double r2 = dx*dx + dy*dy;
                if (r2 < 0.0025) { // если слишком близко — обмен скоростями
                    std::swap(particles[i].vx, particles[j].vx);
                    std::swap(particles[i].vy, particles[j].vy);
                }
            }
        }
    }

    py::array_t<double> get_positions() const {
    // создаём массив с явным указанием формы
    py::array_t<double> arr(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(particles.size()),
            static_cast<py::ssize_t>(2)
        }
    );

    auto buf = arr.mutable_unchecked<2>();
    for (size_t i = 0; i < particles.size(); ++i) {
        buf(i, 0) = particles[i].x;
        buf(i, 1) = particles[i].y;
    }
    return arr;
}

    void init_random() {
        for (auto &p : particles) {
            p.x = 2.0 * L * (rand() / double(RAND_MAX) - 0.5);
            p.y = 2.0 * L * (rand() / double(RAND_MAX) - 0.5);
            p.vx = 0.5 * (rand() / double(RAND_MAX) - 0.5);
            p.vy = 0.5 * (rand() / double(RAND_MAX) - 0.5);
            p.m = 1.0;
        }
    }

private:
    std::vector<Particle> particles;
    double L;
    double dt;
};

PYBIND11_MODULE(core, m) {
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<size_t, double>())
        .def("step", &Simulation::step)
        .def("get_positions", &Simulation::get_positions);
}
